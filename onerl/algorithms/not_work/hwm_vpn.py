from typing import Optional

import torch
from torch import nn

from onerl.algorithms.algorithm import Algorithm
from onerl.utils.batch.cuda import BatchCuda


class HWMVPNAlgorithm(Algorithm):
    def __init__(self,
                 network: dict,
                 env_params: dict,
                 # Exploration
                 exploration_len: int = 10000,
                 noise_scale: float = 0.2,
                 # Encoder
                 encoder_frame_stack: int = 1,
                 # WM
                 horizon: int = 16,
                 gamma: float = 0.99,
                 # learning rates
                 lr_wm: float = 0.001,
                 lr_policy: float = 0.001,
                 lr_q: float = 0.001,
                 # unused
                 batch_size: Optional[int] = None,
                 replay_buffer_size: Optional[int] = None
                 ):
        super().__init__(network, env_params)
        # Exploration
        self.exploration_len = exploration_len
        self.noise_scale = noise_scale
        # Encoder
        self.encoder_frame_stack = encoder_frame_stack
        # WM
        self.horizon = horizon
        self.gamma = gamma
        # learning rate
        self.lr_wm = lr_wm
        self.lr_policy = lr_policy
        self.lr_q = lr_q
        # optimizer
        self.wm_optimizer = torch.optim.Adam(
            sum([list(self.network[k].parameters()) for k in ["wm", "encoder", "reward"]], []),
            lr=self.lr_wm)
        self.policy_optimizer = torch.optim.Adam(
            self.network["policy"].parameters(),
            lr=self.lr_policy)
        self.q_optimizer = torch.optim.Adam(
            self.network["q"].parameters(),
            lr=self.lr_q
        )

    def forward(self, obs: torch.Tensor, ticks: int):
        # full exploration
        if (ticks is not None) and (ticks < self.exploration_len):
            return torch.rand(obs.shape[0], *self.env_params["act_shape"]) * 2 - 1

        with torch.no_grad():
            act = torch.tanh(self.network["policy"](self.network["encoder"](obs)))
        # exploration
        if ticks is not None:
            return act + self.noise_scale * torch.randn(act.shape, device=act.device)
        # clip
        act = torch.clamp(act, -1, 1)
        return act

    def learn(self, batch: BatchCuda):
        # swap axes (N, Time, C) --> (Time, N, C)
        act = torch.transpose(batch.data["act"], 0, 1)
        rew = torch.transpose(batch.data["rew"], 0, 1)
        discount = (1 - torch.transpose(batch.data["done"], 0, 1)) * self.gamma
        # train world model
        n_old = self.encoder_frame_stack
        h0 = self.network["encoder"](batch.data["obs"][:, :n_old])
        h, _ = self.network["wm"](act[(n_old - 1):-1], h0.unsqueeze(0))
        # decode predicted
        h_flat = h.view(-1, *h.shape[2:])
        pred_rew_discount = self.network["reward"](h_flat)
        pred_rew_discount = pred_rew_discount.view(h.shape[0], h.shape[1], -1)
        # loss
        wm_rew_loss = torch.mean((pred_rew_discount[..., 0] - rew[(n_old - 1):-1]) ** 2)
        wm_discount_loss = torch.mean((pred_rew_discount[..., 1] - discount[(n_old - 1):-1]) ** 2)

        wm_loss = wm_rew_loss + wm_discount_loss

        self.wm_optimizer.zero_grad()
        wm_loss.backward()
        self.wm_optimizer.step()

        # train q
        # sample initial states (N, C)
        with torch.no_grad():
            init_state = self.network["encoder"](batch.data["obs"][:, :n_old])
            init_act = batch.data["act"][:, n_old - 1]

            h = init_state
            tot_rew = 0
            tot_discount = 1
            for idx in range(self.horizon):
                a = init_act
                if idx != 0:
                    a = torch.tanh(self.network["policy"](h))
                    # a = torch.clip(a + self.noise_scale * torch.randn(a.shape, device=a.device), -1, 1)  # TPS
                # predict 1 step
                h, _ = self.network["wm"](a.unsqueeze(0), h.unsqueeze(0))
                h = h[0]
                # decode reward
                pred_rew_discount = self.network["reward"](h)
                tot_rew = tot_rew + tot_discount * pred_rew_discount[:, 0]
                tot_discount = tot_discount * pred_rew_discount[:, 1]

        q = self.network["q"](init_state, init_act).squeeze(-1)
        q_loss = torch.mean((q - tot_rew) ** 2)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # train pi
        a = torch.tanh(self.network["policy"](init_state))
        q = self.network["q"](init_state, a)
        policy_loss = -torch.mean(q)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return {
            "update": True,

            "wm_rew_loss": wm_rew_loss.item(),
            "wm_discount_loss": wm_discount_loss.item(),
            "wm_loss": wm_loss.item(),

            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item()
        }

    def policy_state_dict(self):
        policy = self.network["policy"]
        return policy.module.state_dict() if hasattr(policy, "module") else policy.state_dict()
