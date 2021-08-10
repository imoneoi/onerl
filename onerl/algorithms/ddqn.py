from copy import deepcopy

import torch

from onerl.algorithms.algorithm import Algorithm
from onerl.utils.batch.cuda import BatchCuda


class DDQNAlgorithm(Algorithm):
    def __init__(
        self, network: dict, env_params: dict,
        # hyper-parameters
        lr: float,
        gamma: float,
        target_update_freq: int,
        # exploration
        eps_start: float,
        eps_final: float,
        eps_final_steps: float
    ):
        super().__init__(network, env_params)
        # action
        assert "act_n" in self.env_params, "DDQN: Environment actions must be discrete."
        self.act_n = self.env_params["act_n"]
        # hyper-parameters
        self.lr = lr
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        # exploration
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_final_steps = eps_final_steps
        # target network
        self.target_iter = 0
        self.target_network = {k: deepcopy(v) for k, v in self.network.items()}
        for v in self.target_network.values():
            v.eval()
        # optimizer
        self.optimizer = torch.optim.Adam(list(self.network["feature_extractor"].parameters()) +
                                          list(self.network["critic"].parameters()), lr=self.lr)

    def forward(self, obs: torch.Tensor):
        with torch.no_grad():
            feature = self.network["feature_extractor"](obs)
            q = self.network["critic"](feature)

        act = torch.argmax(q, dim=-1)
        return act

    def learn(self, batch: BatchCuda):
        # TODO: WARNING: DistributedDataParallel enabled here
        # next q
        with torch.no_grad():
            next_obs = batch.data["obs"][1:]
            curr_next_q = self.network["critic"](self.network["feature_extractor"](next_obs))
            target_next_q = self.target_network["critic"](self.target_network["feature_extractor"](next_obs))
            next_q = target_next_q[torch.arange(target_next_q.shape[0], device=target_next_q.device),
                                   torch.argmax(curr_next_q, dim=-1)]
        # update target
        update_target = batch.data["rew"][-2] + self.gamma * (1. - batch.data["done"][-2]) * next_q
        # current q
        q = self.network["critic"](self.network["feature_extractor"](batch.data["obs"][:-1]))
        q = q[torch.arange(q.shape[0], device=q.device), batch.data["act"][-2]]
        # back prop
        loss = torch.mean((q - update_target) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def serialize_policy(self):
        # state dict of networks
        return {k: v.state_dict() for k, v in self.network.items()}

    def deserialize_policy(self, data):
        for k, v in self.network.items():
            v.load_state_dict(data[k])
