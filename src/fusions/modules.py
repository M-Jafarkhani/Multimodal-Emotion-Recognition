import torch
from torch import nn
from torch.autograd import Variable


class Concat(nn.Module):
    """Concatenation of input data on dimension 1."""

    def __init__(self):
        """Initialize Concat Module."""
        super(Concat, self).__init__()

    def forward(self, modalities):
        """
        Forward Pass of Concat.

        :param modalities: An iterable of modalities to combine
        """
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)


class ConcatEarly(nn.Module):
    """Concatenation of input data on dimension 2."""

    def __init__(self):
        """Initialize ConcatEarly Module."""
        super(ConcatEarly, self).__init__()

    def forward(self, modalities):
        """
        Forward Pass of ConcatEarly.

        :param modalities: An iterable of modalities to combine
        """
        return torch.cat(modalities, dim=2)


class TensorFusion(nn.Module):
    """
    Implementation of TensorFusion Networks.

    See https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py for more and the original code.
    """

    def __init__(self):
        """Instantiates TensorFusion Network Module."""
        super().__init__()

    def forward(self, modalities):
        """
        Forward Pass of TensorFusion.

        :param modalities: An iterable of modalities to combine.
        """
        if len(modalities) == 1:
            return modalities[0]

        mod0 = modalities[0]
        nonfeature_size = mod0.shape[:-1]

        m = torch.cat(
            (
                Variable(
                    torch.ones(*nonfeature_size, 1)
                    .type(mod0.dtype)
                    .to(mod0.device),
                    requires_grad=False,
                ),
                mod0,
            ),
            dim=-1,
        )
        for mod in modalities[1:]:
            mod = torch.cat(
                (
                    Variable(
                        torch.ones(*nonfeature_size, 1)
                        .type(mod.dtype)
                        .to(mod.device),
                        requires_grad=False,
                    ),
                    mod,
                ),
                dim=-1,
            )
            fused = torch.einsum("...i,...j->...ij", m, mod)
            m = fused.reshape([*nonfeature_size, -1])

        return m


class LowRankTensorFusion(nn.Module):
    def __init__(self, input_dims, output_dim, rank, flatten=True):
        """
        Initialize LowRankTensorFusion object.

        :param input_dims: list or tuple of integers indicating input dimensions of the modalities
        :param output_dim: output dimension
        :param rank: a hyperparameter of LRTF. See link above for details
        :param flatten: Boolean to dictate if output should be flattened or not. Default: True

        """
        super(LowRankTensorFusion, self).__init__()

        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten

        self.factors = []
        for input_dim in input_dims:
            factor = nn.Parameter(
                torch.Tensor(self.rank, input_dim + 1, self.output_dim)
            ).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            nn.init.xavier_normal(factor)
            self.factors.append(factor)

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank)).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim)).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        
        nn.init.xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, modalities):
        """
        Forward Pass of Low-Rank TensorFusion.

        :param modalities: An iterable of modalities to combine.
        """
        batch_size = modalities[0].shape[0]
        fused_tensor = 1
        for modality, factor in zip(modalities, self.factors):
            ones = Variable(
                torch.ones(batch_size, 1).type(modality.dtype),
                requires_grad=False,
            ).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            if self.flatten:
                modality_withones = torch.cat(
                    (ones, torch.flatten(modality, start_dim=1)), dim=1
                )
            else:
                modality_withones = torch.cat((ones, modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = (
            torch.matmul(
                self.fusion_weights, fused_tensor.permute(1, 0, 2)
            ).squeeze()
            + self.fusion_bias
        )
        output = output.view(-1, self.output_dim)
        return output