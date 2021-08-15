from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn

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
        eps_final_steps: float,
        # unused
        batch_size: int,
        replay_buffer_size: int
    ):
        super().__init__(network, env_params)
        # action
        assert "act_n" in self.env_params, "DDQNAlgorithm: Environment actions must be discrete."
        self.act_n = self.env_params["act_n"]
        # hyper-parameters
        self.lr = lr
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        # exploration
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_final_steps = eps_final_steps
        assert self.eps_start >= self.eps_final, "DDQNAlgorithm: eps_start must be larger than eps_final."
        # target network
        self.target_iter = 0
        self.target_network = nn.ModuleDict({k: deepcopy(v) for k, v in self.network.items()})
        self.target_network.train()
        # optimizer
        self.optimizer = torch.optim.Adam(list(self.network["feature_extractor"].parameters()) +
                                          list(self.network["critic"].parameters()), lr=self.lr)

    def train(self, mode: bool = True):
        self.training = mode
        self.network.train(mode)
        return self

    def forward(self, obs: torch.Tensor, ticks: int):
        with torch.no_grad():
            q = self.network["critic"](self.network["feature_extractor"](obs))

        # greedy actions
        act_greedy = torch.argmax(q, dim=-1).cpu()
        # eps-greedy
        if ticks is None:  # no tick, using min eps
            eps = self.eps_final
        else:
            eps = self.eps_start - (self.eps_start - self.eps_final) * min(1.0, ticks / self.eps_final_steps)
        is_rand = torch.rand(act_greedy.shape[0]) < eps
        return torch.where(is_rand, torch.randint(0, self.act_n, (act_greedy.shape[0], )), act_greedy)

    def sync_weight(self):
        self.target_iter += 1
        if (self.target_iter % self.target_update_freq) == 0:
            for k, v in self.target_network.items():
                v.load_state_dict(self.network[k].state_dict())

    def learn(self, batch: BatchCuda):
        # TODO: WARNING: DistributedDataParallel enabled here
        # TODO: prioritized replay
        with torch.no_grad():
            next_obs = batch.data["obs"][:, 1:]
            curr_next_q = self.network["critic"](self.network["feature_extractor"](next_obs))
            target_next_q = self.target_network["critic"](self.target_network["feature_extractor"](next_obs))
            next_q = target_next_q[torch.arange(target_next_q.shape[0], device=target_next_q.device),
                                   torch.argmax(curr_next_q, dim=-1)]
            # update target
            update_target = batch.data["rew"][:, -2] + self.gamma * (1. - batch.data["done"][:, -2]) * next_q

        # current q
        q = self.network["critic"](self.network["feature_extractor"](batch.data["obs"][:, :-1]))
        q = q[torch.arange(q.shape[0], device=q.device), batch.data["act"][:, -2]]
        # q mean (for visualization)
        q_mean = torch.mean(q)
        # back prop
        loss = torch.mean((q - update_target) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target update
        self.sync_weight()

        return {
            "q_loss": loss.item(),
            "q_mean": q_mean.item()
        }

    def policy_state_dict(self):
        # state dict of networks
        result = OrderedDict()
        for net_name, net in self.network.items():
            net_state_dict = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
            result.update(OrderedDict({"{}.{}".format(net_name, k): v
                                       for k, v in net_state_dict.items()}))

        return result
