from typing import Optional

import torch
from torch import nn

from onerl.algorithms.algorithm import Algorithm
from onerl.utils.batch.cuda import BatchCuda


class HWMCVPAlgorithm(Algorithm):
    def __init__(self,
                 network: dict,
                 env_params: dict,
                 # Exploration
                 exploration_len: int = 0,
                 noise_scale: float = 0.1,
                 # WM
                 unroll_steps: int = 10,
                 gamma: float = 0.99,
                 # learning rates
                 lr_wm: float = 0.001,
                 lr_policy: float = 0.0001,
                 # unused
                 batch_size: Optional[int] = None,
                 replay_buffer_size: Optional[int] = None
                 ):
        super().__init__(network, env_params)
        # Exploration
        self.exploration_len = exploration_len
        self.noise_scale = noise_scale
        # WM
        self.unroll_steps = unroll_steps
        self.gamma = gamma
        # optimizer
        self.wm_optimizer = torch.optim.Adam(
            sum([list(self.network[k].parameters()) for k in ["T", "E", "V"]], []),
            lr=lr_wm)
        self.policy_optimizer = torch.optim.Adam(self.network["P"].parameters(), lr=lr_policy)

    def policy_state_dict(self):
        encoder = self.network["E"]
        policy = self.network["P"]
        return encoder.module.state_dict() if hasattr(encoder, "module") else encoder.state_dict() + \
               policy.module.state_dict() if hasattr(policy, "module") else policy.state_dict()

    def forward(self, obs: torch.Tensor, ticks: int):
        # full exploration
        if (ticks is not None) and (ticks < self.exploration_len):
            return torch.rand(obs.shape[0], *self.env_params["act_shape"]) * 2 - 1

        act = torch.tanh(self.network["P"](self.network["E"](obs)))
        # exploration
        if ticks is not None:
            return act + self.noise_scale * torch.randn(act.shape, device=act.device)
        # clip
        act = torch.clamp(act, -1, 1)
        return act

    def learn(self, batch: BatchCuda):
        with torch.no_grad():
            act = torch.transpose(batch.data["act"], 0, 1)
            cum_rew = torch.cumsum(torch.transpose(batch.data["rew"], 0, 1), dim=0)
            next_obs = torch.transpose(batch.data["obs"], 0, 1)[1:]
            # --> FS * N * (*)

            feature_next_obs = self.network["E"](next_obs.reshape(-1, *next_obs.shape[2:]))
            feature_next_obs = feature_next_obs.view(next_obs.shape[0], next_obs.shape[1], *feature_next_obs.shape[1:])

        # update WM
        tot_v_loss = 0
        tot_c_loss = 0

        wm_loss = 0
        h = self.network["E"](batch.data["obs"][:, 0])
        h0 = h.detach()
        for t in range(self.unroll_steps):
            h = self.network["T"](h, act[t])
            v = self.network["V"](h).squeeze(-1)

            loss_v = 0.01 * torch.mean((v - cum_rew[t]) ** 2)
            loss_c = torch.mean(1 - torch.cosine_similarity(h, feature_next_obs[t], dim=-1))
            wm_loss = wm_loss + loss_v + loss_c

            tot_v_loss += loss_v.item()
            tot_c_loss += loss_c.item()

        self.wm_optimizer.zero_grad()
        wm_loss.backward()
        self.wm_optimizer.step()

        # update Policy
        h = h0
        for t in range(self.unroll_steps):
            a = torch.tanh(self.network["P"](h))
            h = self.network["T"](h, a)

        v = self.network["V"](h).squeeze(-1)
        p_loss = -torch.mean(v)

        self.policy_optimizer.zero_grad()
        p_loss.backward()
        self.policy_optimizer.step()

        return {
            "wm_loss": wm_loss.item(),

            "v_loss": tot_v_loss,
            "c_loss": tot_c_loss,

            "p_loss": p_loss.item()
        }
