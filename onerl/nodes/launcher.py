import collections
import multiprocessing as mp
import argparse
from copy import deepcopy

import yaml

from onerl.nodes import Node
from onerl.utils.import_module import get_class_from_str


def deep_update(dst, src):
    for k, v in src.items():
        if (k in dst) and isinstance(dst[k], dict) and isinstance(src[k], collections.abc.Mapping):
            deep_update(dst[k], src[k])
        else:
            dst[k] = src[k]


def get_node_class(node_class, node_params):
    import_name = node_params.get("import", "onerl.nodes")
    return get_class_from_str(import_name, node_class)


def node_worker_(**kwargs):
    node_class_type = get_node_class(kwargs["node_class"], kwargs["node_config"])

    node_inst = node_class_type(**kwargs)
    node_inst.run()


def preprocess_ns_config(ns_config: dict):
    nodes = ns_config.get("nodes", {})
    for node_class, node_params in nodes.items():
        node_num = node_params.get("num", 1)
        get_node_class(node_class, node_params).node_preprocess_ns_config(node_class, node_num, ns_config)


def launch_nodes(yaml_config: dict):
    # preprocess global
    global_ns = yaml_config.get("$global", {}).copy()
    preprocess_ns_config(global_ns)
    global_ns.pop("nodes", None)
    # apply global to all
    all_ns_config = {}
    for ns_name, ns_config in yaml_config.items():
        if ns_name == "$global":
            all_ns_config[ns_name] = ns_config
        else:
            all_ns_config[ns_name] = deepcopy(global_ns)
            deep_update(all_ns_config[ns_name], ns_config)
        preprocess_ns_config(all_ns_config[ns_name])

    # shared objects
    global_objects = {}
    for ns_name, ns_config in all_ns_config.items():
        nodes = ns_config.get("nodes", {})
        for node_class, node_params in nodes.items():
            node_num = node_params.get("num", 1)
            obj_list = get_node_class(node_class, node_params) \
                .node_create_shared_objects(node_class, node_num, ns_config)
            for rank, val in enumerate(obj_list):
                global_objects[Node.get_node_name(ns_name, node_class, rank)] = val

    # launch node processes
    processes = []
    for ns_name, ns_config in all_ns_config.items():
        nodes = ns_config.get("nodes", {})
        for node_class, node_params in nodes.items():
            node_num = node_params.get("num", 1)
            for rank in range(node_num):
                proc = mp.Process(target=node_worker_, name=Node.get_node_name(ns_name, node_class, rank),
                                  kwargs={
                                      "node_class": node_class,
                                      "node_ns": ns_name,
                                      "node_rank": rank,
                                      "node_config": node_params,
                                      "ns_config": ns_config,
                                      "global_objects": global_objects
                                  })
                processes.append(proc)

    # start & join
    [proc.start() for proc in processes]
    [proc.join() for proc in processes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config file")

    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        f.close()

    launch_nodes(config)


if __name__ == "__main__":
    main()
