from collections import OrderedDict

import torch
from torch import nn

from onerl.algorithms.algorithm import Algorithm
from onerl.networks.gpt import TrainerConfig
from onerl.utils.batch.cuda import BatchCuda


class DT2Algorithm(Algorithm):
    def __init__(self,
                 network: dict,
                 env_params: dict,
                 # horizon
                 hist_len: int = 20,
                 pred_len: int = 20,
                 # exploration
                 noise_scale: float = 0.1,
                 # learning rate
                 lr_m: float = 3e-4,
                 lr_p: float = 3e-4,
                 weight_decay: float = 0.1,
                 clip_grad_norm: float = 1.0,
                 # unused
                 batch_size: int = None,
                 replay_buffer_size: int = None
                 ):
        super().__init__(network, env_params)

        self.network = nn.ModuleDict(network)
        self.env_params = env_params
        # horizon
        self.hist_len = hist_len
        self.pred_len = pred_len
        # exploration
        self.noise_scale = noise_scale
        # optimizer
        self.clip_grad_norm = clip_grad_norm
        self.m_optimizer = self.network["M"].configure_optimizers(TrainerConfig(
            learning_rate=lr_m,
            weight_decay=weight_decay
        ))
        self.p_optimizer = self.network["P"].configure_optimizers(TrainerConfig(
            learning_rate=lr_p,
            weight_decay=weight_decay
        ))

    def policy_state_dict(self):
        planner = self.network["P"]
        return planner.module.state_dict() if hasattr(planner, "module") else planner.state_dict()

    def forward(self, obs: torch.Tensor, ticks: int):
        act = torch.tanh(self.network["P"](obs)[:, 0])  # first planned time-step
        # exploration
        if ticks is not None:
            act = act + self.noise_scale * torch.randn_like(act)
        # clip
        act = torch.clip(act, -1, 1)
        return act

    def learn(self, batch: BatchCuda):
        # obs: BSxTxC
        # act: BSxTxC
        # rew: BSxT
        obs_hist = batch.data["obs"][:, :self.hist_len]

        act_full = batch.data["act"]
        act_hist = batch.data["act"][:, :self.hist_len]

        # vp = torch.cumsum(batch.data["rew"], dim=1)

        # train autoregressive M
        # predict
        pred_next_s, pred_vp = self.network["M"](obs_hist, act_full)

        vp = batch.data["rew"] + torch.cat([torch.zeros((pred_vp.shape[0], 1), device=pred_vp.device), pred_vp.detach()[:, :-1]], -1)
        v_weight = torch.arange(vp.shape[1], 0, -1, device=vp.device).unsqueeze(0)

        # loss
        vp_loss = 0.01 * torch.mean((v_weight * (pred_vp - vp)) ** 2)
        s_loss = torch.mean((pred_next_s[:, :-1] - batch.data["obs"][:, 1:]) ** 2)
        tot_loss_m = vp_loss + s_loss
        # optim
        self.network["M"].zero_grad()
        tot_loss_m.backward()
        torch.nn.utils.clip_grad_norm(self.network["M"].parameters(), self.clip_grad_norm)
        self.m_optimizer.step()

        # train planner P
        pred_a = torch.tanh(self.network["P"](obs_hist))
        # target policy smoothing
        pred_a = torch.clip(pred_a + self.noise_scale * torch.randn_like(pred_a), -1, 1)
        # model
        pred_next_s, pred_vp = self.network["M"](obs_hist, torch.cat([act_hist, pred_a], dim=1))
        tot_loss_p = -torch.mean(v_weight[:, self.hist_len:] * pred_vp[:, self.hist_len:])
        # tot_loss_p = -torch.mean(pred_vp[:, -1])
        # optim
        self.network["P"].zero_grad()
        tot_loss_p.backward()
        torch.nn.utils.clip_grad_norm(self.network["P"].parameters(), self.clip_grad_norm)
        self.p_optimizer.step()

        # info
        return {
            "vp_loss": vp_loss.item(),
            "s_loss": s_loss.item(),
            "tot_loss_m": tot_loss_m.item(),

            "tot_loss_p": tot_loss_p.item()
        }
