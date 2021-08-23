from typing import Optional
import argparse
import yaml

import torch
from torch import nn

from onerl.algorithms import Algorithm
from onerl.nodes import OptimizerNode, EnvNode
from onerl.utils.dtype import numpy_to_torch_dtype_dict


class PolicyInferenceWrapper(nn.Module):
    def __init__(self, policy: Algorithm):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor):
        return self.policy(obs, None)


def export_trained_policy(ns_config: dict,
                          load_model_path: str,
                          export_path: Optional[str] = None):
    # preprocess config
    EnvNode.node_preprocess_ns_config("EnvNode", 1, ns_config)
    # create policy from config
    device = torch.device("cpu")
    policy = OptimizerNode.create_algo(ns_config, device)
    # load policy parameters
    policy_state_dict = torch.load(load_model_path, map_location=device)
    policy_state_dict = {k.replace("module.", ""): v
                         for k, v in policy_state_dict.items()}
    policy.load_state_dict(policy_state_dict, strict=False)
    # create wrapper
    policy = PolicyInferenceWrapper(policy)
    # create dummy input
    x = torch.zeros((1, ns_config["env"]["frame_stack"], *ns_config["env"]["obs_shape"]),
                    dtype=numpy_to_torch_dtype_dict[ns_config["env"]["obs_dtype"]])
    # export onnx
    if not export_path:
        export_path = load_model_path + ".onnx"
    torch.onnx.export(policy, x, export_path,
                      input_names=["obs"],
                      output_names=["act"],
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      verbose=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config file")
    parser.add_argument("model_path", type=str, help="Path to trained policy model")
    parser.add_argument("--export_path", type=str, default=None, help="Exported ONNX save path")

    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        f.close()

    export_trained_policy(
        config["$global"],
        args.model_path,
        args.export_path
    )


if __name__ == "__main__":
    main()
