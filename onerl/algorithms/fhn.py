from collections import OrderedDict

import torch

from onerl.algorithms.algorithm import Algorithm
from onerl.utils.batch.cuda import BatchCuda


class FHNAlgorithm(Algorithm):
    def __init__(
        self, network: dict, env_params: dict,
        # hyper-parameters
        lr: float,
        gamma: float,
        h: int,
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
        # exploration
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_final_steps = eps_final_steps
        assert self.eps_start >= self.eps_final, "VSNAlgorithm: eps_start must be larger than eps_final."
        # optimizer
        self.optimizer = torch.optim.Adam(list(self.network["feature_extractor"].parameters()) +
                                          list(self.network["critic"].parameters()), lr=self.lr)

    def policy_state_dict(self):
        # state dict of networks
        result = OrderedDict()
        for net_name, net in self.network.items():
            net_state_dict = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
            result.update(OrderedDict({"{}.{}".format(net_name, k): v
                                       for k, v in net_state_dict.items()}))

        return result

    def _predict_q(self, obs):
        return self.network["critic"](self.network["feature_extractor"](obs)).view(-1, self.act_n, self.h)

    def forward(self, obs: torch.Tensor, ticks: int):
        with torch.no_grad():
            q = self._predict_q(obs)
            loss_weight = torch.arange(self.h, 0, -1, device=q.device)
            u = torch.mean(q * loss_weight.view(1, 1, -1), dim=-1)

        # greedy actions
        act_greedy = torch.argmax(u, dim=-1).cpu()
        # eps-greedy
        if ticks is None:  # no tick, using min eps
            eps = self.eps_final
        else:
            eps = self.eps_start - (self.eps_start - self.eps_final) * min(1.0, ticks / self.eps_final_steps)
        is_rand = torch.rand(act_greedy.shape[0]) < eps
        return torch.where(is_rand, torch.randint(0, self.act_n, (act_greedy.shape[0], )), act_greedy)

    def learn(self, batch: BatchCuda):
        # TODO: WARNING: DistributedDataParallel enabled here
        with torch.no_grad():
            next_q = self._predict_q(batch.data["obs"][:, 1:])
            next_q = torch.max(next_q, dim=-2).values  # max on action
            # update target
            update_target = batch.data["rew"][:, -2].unsqueeze(-1) \
                            + self.gamma * (1. - batch.data["done"][:, -2]).unsqueeze(-1) * torch.cat([torch.zeros(next_q.shape[0], 1, device=next_q.device), next_q[:, :-1]], -1)

        # current w
        q = self._predict_q(batch.data["obs"][:, :-1])
        q = q[torch.arange(q.shape[0], device=q.device), batch.data["act"][:, -2]]
        # w mean (for visualization)
        with torch.no_grad():
            u_mean = torch.mean(torch.mean(q, dim=-1))
        # back prop
        loss_weight = torch.arange(self.h, 0, -1, device=q.device)
        # loss_weight = loss_weight / torch.sum(loss_weight)
        loss = torch.mean(((q - update_target) * loss_weight.unsqueeze(0)) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "w_loss": loss.item(),
            "u_mean": u_mean.item()
        }
