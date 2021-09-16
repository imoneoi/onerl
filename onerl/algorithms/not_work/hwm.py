from typing import Optional
from copy import deepcopy

import torch
from torch import nn

from onerl.algorithms.algorithm import Algorithm
from onerl.utils.batch.cuda import BatchCuda


class HWMAlgorithm(Algorithm):
    def __init__(self,
                 network: dict,
                 env_params: dict,
                 # Exploration
                 exploration_len: int = 0,
                 noise_scale: float = 0.2,
                 noise_max: float = 0.5,
                 # HWM
                 gamma: float = 0.99,
                 # learning rates
                 lr_wm: float = 1e-3,
                 lr_q: float = 1e-3,
                 lr_policy: float = 5e-4,
                 # unused
                 batch_size: Optional[int] = None,
                 replay_buffer_size: Optional[int] = None
                 ):
        super().__init__(network, env_params)
        self.act_max = torch.tensor(self.env_params["act_max"])
        # Exploration
        self.exploration_len = exploration_len
        self.noise_scale = noise_scale
        self.noise_max = noise_max
        # WM
        self.gamma = gamma
        # learning rate
        self.lr_wm = lr_wm
        self.lr_q = lr_q
        self.lr_policy = lr_policy
        # wm
        self.wm = {int(k[len("wm"):]): v for k, v in self.network.items() if k.startswith("wm")}
        self.qa = {int(k[len("qa"):]): v for k, v in self.network.items() if k.startswith("qa")}
        self.qb = {int(k[len("qb"):]): v for k, v in self.network.items() if k.startswith("qb")}
        self.H = len(self.wm)
        # optimizers
        self.wm_optimizers = [torch.optim.Adam(v.parameters(), lr=self.lr_wm) for v in self.wm.values()]
        self.qa_optimizers = [torch.optim.Adam(v.parameters(), lr=self.lr_q) for v in self.qa.values()]
        self.qb_optimizers = [torch.optim.Adam(v.parameters(), lr=self.lr_q) for v in self.qb.values()]
        self.actor_optimizer = torch.optim.Adam(self.network["actor"].parameters(), lr=self.lr_policy)

        # train index
        self._cur_idx = 0

    def forward(self, obs: torch.Tensor, ticks: int):
        # full exploration
        if (ticks is not None) and (ticks < self.exploration_len):
            return self.act_max * (torch.rand(obs.shape[0], *self.env_params["act_shape"]) * 2 - 1)

        act = torch.tanh(self.network["actor"](obs))
        # exploration
        if ticks is not None:
            return act + torch.clip(self.noise_scale * torch.randn(act.shape, device=act.device),
                                    -self.noise_max, self.noise_max)
        # clip
        act = torch.clip(act, -1, 1)
        return act

    def predict(self, k: int, obs: torch.Tensor, act: torch.Tensor,
                predict_next_obs: bool = True,
                return_all_q: bool = False):
        # return rew, discount, next_obs
        next_obs = None
        if predict_next_obs:
            next_obs = self.wm[k](obs, act)

        next_qa_disc = self.qa[k](obs, act)
        next_qb_disc = self.qb[k](obs, act)
        if return_all_q:
            return next_obs, \
                   (next_qa_disc[:, 0], next_qb_disc[:, 0]),\
                   (next_qa_disc[:, 1], next_qb_disc[:, 1])

        return next_obs, \
               torch.min(next_qa_disc[:, 0], next_qb_disc[:, 0]), \
               torch.min(next_qa_disc[:, 1], next_qb_disc[:, 1])

    def learn(self, batch: BatchCuda):
        metrics = {}

        if self._cur_idx == self.H:
            # actor step
            obs = batch.data["obs"][:, 0]
            act = self(obs, None)

            sum_q = 0
            for k in range(self.H):
                _, pred_q, pred_disc = self.predict(k, obs, act, predict_next_obs=False)
                sum_q = sum_q + pred_q

            loss = -torch.mean(sum_q)
            # _, pred_q, pred_disc = self.predict(self.H - 1, obs, act, predict_next_obs=False)
            # loss = -torch.mean(pred_q)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            metrics["actor_loss"] = loss.item()
        else:
            # wm step
            obs = batch.data["obs"][:, 0]
            act = batch.data["act"][:, 0]
            if self._cur_idx == 0:
                target_q = batch.data["rew"][:, 0]
                target_disc = self.gamma * (1 - batch.data["done"][:, 0])
                target_next_obs = batch.data["obs"][:, 1]
            else:
                with torch.no_grad():
                    pred_k_obs, pred_k_q, pred_k_disc = self.predict(self._cur_idx - 1, obs, act)
                    pred_2k_obs, pred_2k_q, pred_2k_disc = \
                        self.predict(self._cur_idx - 1, pred_k_obs, self(pred_k_obs, None))

                    target_q = pred_k_q + pred_k_disc * pred_2k_q
                    target_disc = pred_k_disc * pred_2k_disc
                    target_next_obs = pred_2k_obs

            # optimize
            cur_next_obs, (cur_qa_q, cur_qb_q), (cur_qa_disc, cur_qb_disc) = self.predict(self._cur_idx, obs, act,
                                                                                          return_all_q=True)

            loss_fn = torch.nn.MSELoss()
            loss_qa = loss_fn(cur_qa_q, target_q) + loss_fn(cur_qa_disc, target_disc)
            loss_qb = loss_fn(cur_qb_q, target_q) + loss_fn(cur_qb_disc, target_disc)
            loss_next_obs = loss_fn(cur_next_obs, target_next_obs)

            self.qa_optimizers[self._cur_idx].zero_grad()
            loss_qa.backward()
            self.qa_optimizers[self._cur_idx].step()

            self.qb_optimizers[self._cur_idx].zero_grad()
            loss_qb.backward()
            self.qb_optimizers[self._cur_idx].step()

            self.wm_optimizers[self._cur_idx].zero_grad()
            loss_next_obs.backward()
            self.wm_optimizers[self._cur_idx].step()

            metrics["qa{}_loss".format(self._cur_idx)] = loss_qa.item()
            metrics["qb{}_loss".format(self._cur_idx)] = loss_qb.item()
            metrics["wm{}_loss".format(self._cur_idx)] = loss_next_obs.item()

        self._cur_idx = (self._cur_idx + 1) % (self.H + 1)

        return metrics

    def policy_state_dict(self):
        policy = self.network["actor"]
        return policy.module.state_dict() if hasattr(policy, "module") else policy.state_dict()
