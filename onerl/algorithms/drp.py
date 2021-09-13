from typing import Optional

import torch
from torch import nn

from onerl.algorithms.algorithm import Algorithm
from onerl.utils.batch.cuda import BatchCuda


class DRPAlgorithm(Algorithm):
    # deep residual planning
    def __init__(self,
                 network: dict,
                 env_params: dict,
                 # Exploration
                 noise_scale: float = 0.2,
                 exploration_len: int = 0,
                 # WM
                 horizon: int = 16,
                 gamma: float = 0.99,
                 policy_delay: int = 1,
                 # learning rates
                 lr_wm: float = 0.001,
                 lr_policy: float = 0.001,
                 # unused
                 batch_size: Optional[int] = None,
                 replay_buffer_size: Optional[int] = None
                 ):
        super().__init__(network, env_params)
        # Exploration
        self.noise_scale = noise_scale
        self.exploration_len = exploration_len
        # WM
        self.horizon = horizon
        self.gamma = gamma
        # policy delay
        self._iter = 0
        self.policy_delay = policy_delay
        # learning rate
        self.lr_wm = lr_wm
        self.lr_policy = lr_policy
        # optimizer
        self.wm_optimizer = torch.optim.Adam(self.network["wm"].parameters(), lr=self.lr_wm)
        self.r_optimizer = torch.optim.Adam(self.network["r"].parameters(), lr=self.lr_wm)
        self.policy_optimizer = torch.optim.Adam(self.network["policy"].parameters(), lr=self.lr_policy)

    def forward(self, obs: torch.Tensor, ticks: int):
        # random exploration
        if (ticks is not None) and (ticks < self.exploration_len):
            return torch.rand(self.env_params["act_shape"]) * 2 - 1

        act = torch.tanh(self.network["policy"](obs))
        # exploration
        if ticks is not None:
            return act + self.noise_scale * torch.randn(act.shape, device=act.device)
        # clip
        act = torch.clamp(act, -1, 1)
        return act

    def policy_state_dict(self):
        policy = self.network["policy"]
        return policy.module.state_dict() if hasattr(policy, "module") else policy.state_dict()

    def learn(self, batch: BatchCuda):
        metrics = {}
        # optimize WM
        obs = batch.data["obs"][:, 0]
        act = batch.data["act"][:, 0]
        rew = batch.data["rew"][:, 0]
        next_obs = batch.data["obs"][:, 1]

        pred_res_next_obs = self.network["wm"](obs, act)
        pred_rew = self.network["r"](obs, act).squeeze(-1)
        loss_fn = torch.nn.MSELoss()
        wm_loss = loss_fn(pred_res_next_obs, next_obs - obs)
        r_loss = loss_fn(pred_rew, rew)

        self.wm_optimizer.zero_grad()
        wm_loss.backward()
        self.wm_optimizer.step()
        self.r_optimizer.zero_grad()
        r_loss.backward()
        self.r_optimizer.step()

        metrics["wm_loss"] = wm_loss.item()
        metrics["r_loss"] = r_loss.item()

        # optimize policy
        self._iter += 1
        if (self._iter % self.policy_delay) == 0:
            sum_rewards = 0
            discount = 1

            pred_obs = obs
            for _ in range(self.horizon):
                pred_act = self(pred_obs, None)
                pred_rew = self.network["r"](pred_obs, pred_act).squeeze(-1)
                pred_res_next_obs = self.network["wm"](pred_obs, pred_act)
                pred_obs = pred_obs + pred_res_next_obs

                sum_rewards = sum_rewards + discount * pred_rew
                discount *= self.gamma

            policy_loss = -torch.mean(sum_rewards)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            metrics["policy_loss"] = policy_loss.item()

        return metrics
