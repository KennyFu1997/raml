import torch

# From https://github.com/jhgan00/image-retrieval-transformers
from lib.layers.functional import chamfer_sim


class DifferentialEntropyRegularization(torch.nn.Module):

    def __init__(self, weight, eps=1e-8):
        super(DifferentialEntropyRegularization, self).__init__()
        self.eps = eps
        self.pdist = torch.nn.PairwiseDistance(2)
        self.weight = weight

    def forward(self, x):

        with torch.no_grad():
            dots = x @ x.t()
            # dots = chamfer_sim(x, x)
            n = x.shape[0]
            dots.view(-1)[::(n + 1)].fill_(-1)  # trick to fill diagonal with -1
            _, I = torch.max(dots, 1)  # max inner prod -> min distance

        rho = self.pdist(x, x[I])

        # dist_matrix = torch.norm(x.unsqueeze(1) - x.unsqueeze(0), p=2, dim=-1)
        # rho = dist_matrix.topk(k=2, largest=False)[0][:, 1]

        loss = -torch.log(rho + self.eps).mean()

        return self.weight * loss