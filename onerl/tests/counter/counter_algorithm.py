import torch

from onerl.utils.batch.cuda import BatchCuda
from onerl.algorithms import Algorithm


class CounterAlgorithm(Algorithm):
    def __init__(self,
                 network: dict,
                 env_params: dict,
                 **kwargs):
        super().__init__(network, env_params)
        self.ticks = torch.nn.Parameter(torch.zeros((), dtype=torch.int64), requires_grad=False)

    def forward(self, obs: torch.Tensor, ticks: int):
        # Assertion 1: N_BS * N_FS * 2
        assert len(obs.shape) == 3
        assert obs.shape[2] == 2
        # Assertion 2: Good frame stacking
        # TODO

        return torch.tile(self.ticks * 10, [obs.shape[0]]) + (obs[:, -1, 0] % 10)

    def learn(self, batch: BatchCuda):
        # add tick
        self.ticks.add_(1)

        # N_BS * N_FS * 2
        # Assertion 1: good frame stacking
        obs = batch.data["obs"]
        assert (obs[:, :, 0] - torch.arange(obs.shape[1], device=obs.device).view(1, -1) == obs[:, 0, 0].view(-1, 1)) \
               .all()

        # Assertion 2: good action recording
        if not (obs[:, 1:, 1] == torch.where(batch.data["done"][:, :-1] == 1, -1, batch.data["act"][:, :-1])).all():
            assert False

        # Assertion 3: good done recording
        assert ((batch.data["done"][:, :-1] == 1) == (batch.data["obs"][:, 1:, 1] == -1)).all()
        return {"update": 1}

    def serialize_policy(self):
        return self.state_dict()

    def deserialize_policy(self, data):
        self.load_state_dict(data)
