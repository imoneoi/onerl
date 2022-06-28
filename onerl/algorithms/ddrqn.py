from typing import Optional
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from onerl.algorithms.algorithm import Algorithm
from onerl.utils.batch.cuda import BatchCuda


@torch.jit.script
def vf_scale_h(x):
    # Value function rescaling for reducing variance
    # https://arxiv.org/pdf/1805.11593.pdf

    # WARNING! Rescaled Bellman operator may not yield same optimal policy on stochastic environments!

    eps = 1e-3
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


@torch.jit.script
def vf_scale_hinv(x):
    eps = 1e-3
    return ((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1


class DDRQNAlgorithm(Algorithm):
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
        # vf rescale
        vf_rescale: bool = True,
        # drqn
        warmup_timesteps: Optional[int] = None,
        # unused
        batch_size: Optional[int] = None,
        replay_buffer_size: Optional[int] = None
    ):
        super().__init__(network, env_params)
        # action
        assert "act_n" in self.env_params, "DDRQNAlgorithm: Environment actions must be discrete."
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
        # vf rescale
        self.vf_rescale = vf_rescale
        # drqn
        self.warmup_timesteps = warmup_timesteps
        # target network
        self.target_iter = 0
        self.target_critic = deepcopy(self.network["critic"])
        self.target_critic.train()  # target network always in training
        # optimizer
        self.optimizer = torch.optim.Adam(list(self.network["feature_extractor"].parameters()) +
                                          list(self.network["recurrent"].parameters()) +
                                          list(self.network["critic"].parameters()), lr=self.lr)
        self.network["recurrent"].train()  # recurrent module always in training (for dropout!)

    def train(self, mode: bool = True):
        # target network & recurrent module always in training mode
        self.training = mode

        self.network["feature_extractor"].train(mode)
        self.network["critic"].train(mode)
        return self

    def recurrent_state(self):
        return self.network["recurrent"].recurrent_state_shape()

    def _get_obs_features(self, obs: torch.Tensor):
        # obs: (BS, T, C, H, W) --> (BSxT, C, H, W)
        # use reshape here, as obs may not be contigous
        obs_features = self.network["feature_extractor"](obs.reshape(-1, *obs.shape[2:]))
        # restore shape: (BSxT, C) --> (BS, T, C)
        return obs_features.view(obs.shape[0], obs.shape[1], obs_features.shape[1])

    def forward(self, obs: torch.Tensor, ticks: int, rstate: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            obs_feature = self._get_obs_features(obs)
            _, next_rstate = self.network["recurrent"](obs_feature, rstate)

            q = self.network["critic"](next_rstate)
            # greedy actions
            act_greedy = torch.argmax(q, dim=-1)

        # eps-greedy
        if ticks is None:  # no tick, using min eps
            eps = self.eps_final
        else:
            eps = self.eps_start - (self.eps_start - self.eps_final) * min(1.0, ticks / self.eps_final_steps)

        is_rand  = torch.rand(act_greedy.shape[0], device=act_greedy.device) < eps
        act_rand = torch.randint(0, self.act_n, (act_greedy.shape[0], ), device=act_greedy.device)

        act = torch.where(is_rand, act_rand, act_greedy)

        return act, next_rstate

    def policy_state_dict(self) -> OrderedDict:
        # state dict of networks
        result = OrderedDict()
        for net_name, net in self.network.items():
            net_state_dict = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
            result.update(OrderedDict({"{}.{}".format(net_name, k): v
                                       for k, v in net_state_dict.items()}))

        return result

    def _sync_weight(self):
        self.target_iter += 1
        if (self.target_iter % self.target_update_freq) == 0:
            self.target_critic.load_state_dict(self.network["critic"].state_dict())

    def learn(self, batch: BatchCuda, ticks: int) -> dict:
        # TODO: WARNING: DistributedDataParallel enabled here
        # TODO: prioritized replay

        # obs: (BS, FS, C)
        # rstate: (BS, C)

        # mask: do not supervise values after first episode termination
        # mask: 0 after first episode done, 1 otherwise
        # mask = torch.cumprod(1 - F.pad(batch.done[:, :-2], (1, 0)), -1)

        # warmup (update rstate)
        num_warmup = self.warmup_timesteps if self.warmup_timesteps is not None else 0
        if num_warmup > 0:
            with torch.no_grad():
                obs_feature = self._get_obs_features(batch.obs[:, :num_warmup])
                _, rstate = self.network["recurrent"](obs_feature, batch.rstate)

                obs = batch.obs[:, num_warmup:]
                act = batch.act[:, num_warmup:]
                rew = batch.rew[:, num_warmup:]
                done = batch.done[:, num_warmup:]
                # mask = mask[:, num_warmup:]
        else:
            obs = batch.obs
            act = batch.act
            rew = batch.rew
            done = batch.done

            rstate = batch.rstate

        # bptt & q-loss
        obs_feature = self._get_obs_features(obs)
        h_out, _ = self.network["recurrent"](obs_feature, rstate)
        q = self.network["critic"](h_out)
        with torch.no_grad():
            q_target_critic = self.target_critic(h_out)
            # select action using Q, evaluate Q-value using Q^target
            # q_target: BS x T
            q_target = q_target_critic.gather(-1, q.argmax(-1, keepdim=True)).squeeze(-1)

            # Bellman target
            if self.vf_rescale:
                q_target = vf_scale_h(rew[:, :-1] + self.gamma * (1 - done[:, :-1]) * vf_scale_hinv(q_target[:, 1:]))
            else:
                q_target = rew[:, :-1] + self.gamma * (1 - done[:, :-1]) * q_target[:, 1:]

        q_curr = q.gather(-1, act.unsqueeze(-1)).squeeze(-1)
        q_curr = q_curr[:, :-1]

        # loss = torch.mean(mask * ((q_target - q_curr) ** 2))
        loss = torch.mean((q_target - q_curr) ** 2)
        q_mean = torch.mean(q_curr)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target update
        self._sync_weight()

        return {
            "q_loss": loss.item(),
            "q_mean": q_mean.item()
        }
