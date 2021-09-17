from copy import deepcopy
from collections import OrderedDict

import torch
from torch import nn

from onerl.algorithms.algorithm import Algorithm
from onerl.utils.batch.cuda import BatchCuda


class VSNAlgorithm(Algorithm):
    """
    Value Sequence Networks
    ( implementation based on DQN )

    W-function: max expected reward
    U-function: upper bound of Q
    output: (BS, NAct, T)
    """
    def __init__(
        self, network: dict, env_params: dict,
        # hyper-parameters
        lr: float,
        gamma: float,
        h: int,
        # target_update_freq: int,
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
        assert "act_n" in self.env_params, "VSNAlgorithm: Environment actions must be discrete."
        self.act_n = self.env_params["act_n"]
        # hyper-parameters
        self.lr = lr
        self.gamma = gamma
        self.h = h
        # self.target_update_freq = target_update_freq
        # exploration
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_final_steps = eps_final_steps
        assert self.eps_start >= self.eps_final, "VSNAlgorithm: eps_start must be larger than eps_final."
        # target network
        # self.target_iter = 0
        # self.target_network = nn.ModuleDict({k: deepcopy(v) for k, v in self.network.items()})
        # self.target_network.train()
        # optimizer
        self.optimizer = torch.optim.Adam(list(self.network["feature_extractor"].parameters()) +
                                          list(self.network["critic"].parameters()), lr=self.lr)

    # def train(self, mode: bool = True):
    #     self.training = mode
    #     self.network.train(mode)
    #     return self

    def policy_state_dict(self):
        # state dict of networks
        result = OrderedDict()
        for net_name, net in self.network.items():
            net_state_dict = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
            result.update(OrderedDict({"{}.{}".format(net_name, k): v
                                       for k, v in net_state_dict.items()}))

        return result

    def _predict_w(self, obs):
        return self.network["critic"](self.network["feature_extractor"](obs)).view(-1, self.act_n, self.h)

    # def _predict_target_w(self, obs):
    #     return self.target_network["critic"](self.target_network["feature_extractor"](obs)).view(-1, self.act_n, self.h)

    def forward(self, obs: torch.Tensor, ticks: int):
        with torch.no_grad():
            u = torch.mean(self._predict_w(obs), dim=-1)

        # greedy actions
        act_greedy = torch.argmax(u, dim=-1).cpu()
        # eps-greedy
        if ticks is None:  # no tick, using min eps
            eps = self.eps_final
        else:
            eps = self.eps_start - (self.eps_start - self.eps_final) * min(1.0, ticks / self.eps_final_steps)
        is_rand = torch.rand(act_greedy.shape[0]) < eps
        return torch.where(is_rand, torch.randint(0, self.act_n, (act_greedy.shape[0], )), act_greedy)

    # def sync_weight(self):
    #     self.target_iter += 1
    #     if (self.target_iter % self.target_update_freq) == 0:
    #         for k, v in self.target_network.items():
    #             v.load_state_dict(self.network[k].state_dict())

    def learn(self, batch: BatchCuda):
        # TODO: WARNING: DistributedDataParallel enabled here
        with torch.no_grad():
            next_w = self._predict_w(batch.data["obs"][:, 1:])
            next_w = torch.max(next_w, dim=-2).values  # max on action
            # next_w = torch.clip(next_w, -1, 1)   # we know rew >= -1 && rew <= 1
            # update target
            update_target = batch.data["rew"][:, -2].unsqueeze(-1) \
                            + self.gamma * (1. - batch.data["done"][:, -2]).unsqueeze(-1) * torch.cat([torch.zeros(next_w.shape[0], 1, device=next_w.device), next_w[:, :-1]], -1)

        # current w
        w = self._predict_w(batch.data["obs"][:, :-1])
        w = w[torch.arange(w.shape[0], device=w.device), batch.data["act"][:, -2]]
        # w mean (for visualization)
        with torch.no_grad():
            u_mean = torch.mean(torch.mean(w, dim=-1))
        # back prop
        loss_weight = torch.arange(self.h, 0, -1, device=w.device)
        loss_weight = loss_weight / torch.sum(loss_weight)
        loss = torch.mean(((w - update_target) ** 2) * loss_weight.unsqueeze(0))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # sync
        # self.sync_weight()

        return {
            "w_loss": loss.item(),
            "u_mean": u_mean.item()
        }
