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
                 exploration_len: int = 10000,
                 noise_scale: float = 0.2,
                 noise_max: float = 0.5,
                 # HWM
                 gamma: float = 0.99,
                 # learning rates
                 lr_wm: float = 1e-3,
                 lr_policy: float = 1e-3,
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
        self.lr_policy = lr_policy
        # wm
        self.wm = {int(k[len("wm"):]): v for k, v in self.network.items() if k.startswith("wm")}
        # optimizers
        self.wm_optimizers = [torch.optim.Adam(v.parameters(), lr=self.lr_wm) for v in self.wm.values()]
        self.actor_optimizer = torch.optim.Adam(self.network["actor"].parameters(), lr=self.lr_policy)

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

    def predict(self, k: int, obs: torch.Tensor, act: torch.Tensor):
        # return rew, discount, next_obs
        pred = self.wm[k](obs, act)
        return pred[:, 0], pred[:, 1], pred[:, 2:]

    def learn(self, batch: BatchCuda):
        obs = batch.data["obs"][:, 0]
        act = batch.data["act"][:, 0]

        # bootstrap WM sequentially
        loss_fn = torch.nn.MSELoss()

        wm_losses = []
        pred_rew = pred_discount = pred_next_obs = None
        for k in range(len(self.wm)):
            if k == 0:
                target_rew = batch.data["rew"][:, 0]
                target_discount = self.gamma * (1 - batch.data["done"][:, 0])
                target_next_obs = batch.data["obs"][:, 1]
            else:
                # using k - 1 prediction
                with torch.no_grad():
                    pred_rew_2, pred_discount_2, pred_next_obs_2 = \
                        self.predict(k - 1, pred_next_obs, self(pred_next_obs, int(1e9)))  # target policy smoothing

                    target_rew = pred_rew + pred_discount * pred_rew_2
                    target_discount = pred_discount * pred_discount_2
                    target_next_obs = pred_next_obs_2

            pred_rew, pred_discount, pred_next_obs = self.predict(k, obs, act)

            rew_loss = loss_fn(pred_rew, target_rew)
            discount_loss = loss_fn(pred_discount, target_discount)
            next_obs_loss = loss_fn(pred_next_obs, target_next_obs)
            loss = rew_loss + discount_loss + next_obs_loss

            self.wm_optimizers[k].zero_grad()
            loss.backward()
            self.wm_optimizers[k].step()

            wm_losses.append(loss.item())

        # train actor using mean of all WM
        actor_output = self(obs, None)
        # mean of all critics
        actor_tot_rew = 0
        for k in range(len(self.wm)):
            pred_rew, pred_discount, pred_next_obs = self.predict(k, obs, actor_output)
            actor_tot_rew = actor_tot_rew + pred_rew

        actor_loss = -torch.mean(actor_tot_rew)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # output metrics
        metrics = {"wm_loss_{}".format(idx): v for idx, v in enumerate(wm_losses)}
        metrics["actor_loss"] = actor_loss.item()
        return metrics

    def policy_state_dict(self):
        policy = self.network["actor"]
        return policy.module.state_dict() if hasattr(policy, "module") else policy.state_dict()
