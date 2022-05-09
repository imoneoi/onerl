from .gpt import GPTConfig, GPT

import torch
from torch import nn


class DecisionGPT(GPT):
    def __init__(
            self,
            # dimensions
            type: str,  # "M" for model, "P" for planner
            state_dims: int,
            action_dims: int,
            block_size: int,
            # transformer parameters
            n_embd: int = 256,
            n_layer: int = 6,
            n_head: int = 4
    ):
        super().__init__(GPTConfig(
            block_size=block_size,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            is_cross_attn=(type == "P")  # planner uses cross attention
        ))
        self.type = type

        # embedding & projection
        self.s_embedding = nn.Sequential(
            nn.Linear(state_dims, n_embd),
            nn.Tanh()
        )
        if type == "M":
            self.a_embedding = nn.Sequential(
                nn.Linear(action_dims, n_embd),
                nn.Tanh()
            )

            self.s_proj = nn.Linear(n_embd, state_dims)
            self.r_proj = nn.Linear(n_embd, 1)
        elif type == "P":
            self.a_proj = nn.Linear(n_embd, action_dims)
        else:
            assert False, "Type must be M (for model) or P (for planner)"

    def forward(self, state, action=None):
        # embedding
        s_embed = self.s_embedding(state)
        # zero pad s_embed
        if self.type == "M":
            a_embed = self.a_embedding(action)

            B, Ts, C = s_embed.shape
            _, Ta, _ = a_embed.shape
            s_embed = torch.cat([s_embed,
                                 torch.zeros((B, Ta - Ts, C), dtype=s_embed.dtype, device=s_embed.device)], dim=1)
            input_embed = s_embed + a_embed
        else:
            input_embed = s_embed

        # forward
        x = super().forward(input_embed)
        # projection
        if self.type == "M":
            return self.s_proj(x), self.r_proj(x).squeeze(-1)
        else:
            return self.a_proj(x)
