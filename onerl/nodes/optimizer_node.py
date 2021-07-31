import torch

from onerl.nodes.node import Node


class OptimizerNode(Node):
    @staticmethod
    def create_model(global_config):
        return torch.nn.Module()

    def run(self):
        self.dummy_init()
