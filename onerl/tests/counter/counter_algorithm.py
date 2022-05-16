import torch

from onerl.tests.test_assert import test_assert

from onerl.utils.batch.cuda import BatchCuda
from onerl.algorithms import Algorithm


class CounterAlgorithm(Algorithm):
    def __init__(self,
                 network: dict,
                 env_params: dict,
                 **kwargs):
        super().__init__(network, env_params)

        self.ticks = torch.nn.Parameter(torch.zeros((), dtype=torch.int64), requires_grad=False)
        self.env_ticks_last_ = 0

    def forward(self, obs: torch.Tensor, ticks: int):
        # Assertion 1: N_BS * N_FS * 2
        test_assert((len(obs.shape) == 3) and (obs.shape[-1] == 2), "Bad observation shape in forward(), expected N_BS * N_FS * 2")
        # Assertion 2: Good frame stacking
        test_assert((((obs[:, -1, 0].view(-1, 1) - torch.arange(obs.shape[1] - 1, -1, -1, device=obs.device).view(1, -1)) == obs[:, :, 0]) | (obs[:, :, 0] == 0)).all(), \
               "Bad frame stacking in forward()")

        return torch.tile(self.ticks * 10, [obs.shape[0]]) + (obs[:, -1, 0] % 10)

    def learn(self, batch: BatchCuda, ticks: int):
        # Assertion 1: Correct env ticks
        if ticks is not None:
            test_assert(ticks >= self.env_ticks_last_, "Bad ticks in learn()")

            self.env_ticks_last_ = ticks

        # add tick
        self.ticks.add_(1)

        # N_BS * N_FS * 2
        # Assertion 2: good frame stacking
        obs = batch.data["obs"]
        test_assert(((obs[:, -1, 0].view(-1, 1) - torch.arange(obs.shape[1] - 1, -1, -1, device=obs.device).view(1, -1)) == obs[:, :, 0]).all(), \
               "Bad frame stacking in learn()")

        # Assertion 3: good action recording
        test_assert((obs[:, 1:, 1] == torch.where(batch.data["done"][:, :-1] == 1, -1, batch.data["act"][:, :-1])).all(), \
               "Bad action recording in learn()")

        # Assertion 4: good reward recording
        test_assert((batch.data["rew"] == batch.data["act"]).all(), \
               "Bad reward recording in learn()")

        # Assertion 5: good done recording
        test_assert(((batch.data["done"][:, :-1] == 1) == (batch.data["obs"][:, 1:, 1] == -1)).all(), \
               "Done flag not correct in learn()")

        return {}

    def policy_state_dict(self):
        return self.state_dict()
