import multiprocessing as mp
import argparse

import yaml

from onerl.nodes import Node
from onerl.utils.import_module import get_class_from_str


def node_worker_(node_class, *args):
    node_class_type = get_class_from_str("onerl.nodes", node_class)

    node_inst = node_class_type(node_class, *args)
    node_inst.init()
    node_inst.run()


def launch_nodes(config: dict):
    nodes = config["nodes"]
    # create class inst
    class_table = {node_class: get_class_from_str("onerl.nodes", node_class) for node_class in nodes.keys()}
    # global config
    global_config = config.copy()
    for node_class, node_params in nodes.items():
        node_num = node_params.get("num", 1)
        class_table[node_class].node_preprocess_global_config(node_class, node_num, global_config)
    # shared objects
    global_objects = {}
    for node_class, node_params in nodes.items():
        node_num = node_params.get("num", 1)
        obj_list = class_table[node_class].node_create_shared_objects(node_class, node_num, global_config)
        for rank, val in enumerate(obj_list):
            global_objects[Node.get_node_name(node_class, rank)] = val
    # launch node processes
    processes = []
    for node_class, node_params in nodes.items():
        node_num = node_params.get("num", 1)
        for rank in range(node_num):
            proc = mp.Process(target=node_worker_, name=Node.get_node_name(node_class, rank),
                              args=(node_class, rank, node_params, global_config, global_objects))
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
