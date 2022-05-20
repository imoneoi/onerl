import collections
import multiprocessing as mp
import argparse
from copy import deepcopy
import time
import re

import yaml

from onerl.nodes.node import Node
from onerl.utils.import_module import get_class_from_str


def deep_update(dst, src):
    for k, v in src.items():
        if (k in dst) and isinstance(dst[k], dict) and isinstance(src[k], collections.abc.Mapping):
            deep_update(dst[k], src[k])
        else:
            dst[k] = src[k]


def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def get_node_class(node_class, node_params):
    import_name = node_params.get("import", "onerl.nodes." + camel_to_snake(node_class))
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
    # spawn mode for multiprocessing
    mp.set_start_method("spawn")

    # preprocess global
    global_ns_config = yaml_config.get("$global", {}).copy()
    preprocess_ns_config(global_ns_config)
    global_ns_config.pop("nodes", None)

    # put global to all ns
    all_ns_config = {}
    for ns_name, ns_config in yaml_config.items():
        if ns_name == "$global":
            all_ns_config[ns_name] = ns_config
        else:
            all_ns_config[ns_name] = deepcopy(global_ns_config)
            deep_update(all_ns_config[ns_name], ns_config)

        preprocess_ns_config(all_ns_config[ns_name])

    # shared objects
    all_ns_objects = {}
    for ns_name, ns_config in all_ns_config.items():
        all_ns_objects[ns_name] = {}

        nodes = ns_config.get("nodes", {})
        for node_class, node_params in nodes.items():
            node_num = node_params.get("num", 1)
            obj_list = get_node_class(node_class, node_params) \
                .node_create_shared_objects(node_class, node_num, ns_config)

            all_ns_objects[ns_name][node_class] = obj_list

    # put global objects to all ns
    global_ns_objects = all_ns_objects.get("$global", {})
    for ns_name, objects in all_ns_objects.items():
        for k, v in global_ns_objects.items():
            if k not in objects:
                objects[k] = v

    # import peer objects
    all_imported_peer_objects = {}
    for ns_name, ns_config in all_ns_config.items():
        all_imported_peer_objects[ns_name] = {}

        nodes = ns_config.get("nodes", {})
        for node_class, node_params in nodes.items():
            node_num = node_params.get("num", 1)
            import_obj_list = get_node_class(node_class, node_params) \
                .node_import_peer_objects(node_class, node_num, ns_config, all_ns_objects[ns_name], all_ns_objects)

            all_imported_peer_objects[ns_name][node_class] = import_obj_list

    # launch node processes
    processes = []
    for ns_name, ns_config in all_ns_config.items():
        nodes = ns_config.get("nodes", {})
        for node_class, node_params in nodes.items():
            node_num = node_params.get("num", 1)
            for rank in range(node_num):
                proc_args = {"target": node_worker_,
                             "name": Node.get_node_name(ns_name, node_class, rank),
                             "kwargs": {
                                   "node_class": node_class,
                                   "node_ns": ns_name,
                                   "node_rank": rank,
                                   "node_config": node_params,
                                   "ns_config": ns_config,

                                   "peer_objects": all_imported_peer_objects[ns_name][node_class][rank]
                             }}
                processes.append([mp.Process(**proc_args), proc_args])

    # start & join
    [proc.start() for proc, _ in processes]

    # node check & auto-recovery
    while True:
        # check process status
        for proc_args in processes:
            if not proc_args[0].is_alive():
                print("Node {} died, respawning".format(proc_args[1]["name"]))

                process = mp.Process(**proc_args[1])
                process.start()
                proc_args[0] = process

        time.sleep(1)


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
