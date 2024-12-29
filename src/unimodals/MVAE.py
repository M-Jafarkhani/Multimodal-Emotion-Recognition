"""Implements various encoders and decoders for MVAE."""

import torch
from torch import nn


class TSEncoder(torch.nn.Module):
    """Implements a time series encoder for MVAE."""

    def __init__(
        self,
        indim,
        outdim,
        finaldim,
        timestep,
        returnvar=True,
        batch_first=False,
    ):
        """Instantiate TSEncoder Module.

        Args:
            indim (int): Input Dimension of GRU
            outdim (int): Output dimension of GRU
            finaldim (int): Output dimension of TSEncoder
            timestep (float): Number of timestamps
            returnvar (bool, optional): Whether to return the output split with the first encoded portion and the next or not. Defaults to True.
            batch_first (bool, optional): Whether the batching dimension is the first dimension of the input or not. Defaults to False.
        """
        super(TSEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=indim, hidden_size=outdim, batch_first=batch_first
        )
        self.indim = indim
        self.ts = timestep
        self.finaldim = finaldim
        if returnvar:
            self.linear = nn.Linear(outdim * timestep, 2 * finaldim)
        else:
            self.linear = nn.Linear(outdim * timestep, finaldim)
        self.returnvar = returnvar

    def forward(self, x):
        """Apply TS Encoder to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        batch = len(x)
        input = x.reshape(batch, self.ts, self.indim).transpose(0, 1)
        output = self.gru(input)[0].transpose(0, 1)
        output = self.linear(output.flatten(start_dim=1))
        if self.returnvar:
            return output[:, : self.finaldim], output[:, self.finaldim :]
        return output


class TSDecoder(torch.nn.Module):
    """Implements a time-series decoder for MVAE."""

    def __init__(self, indim, finaldim, timestep):
        """Instantiate TSDecoder Module.

        Args:
            indim (int): Input dimension
            finaldim (int): Hidden dimension
            timestep (int): Number of timesteps
        """
        super(TSDecoder, self).__init__()
        self.gru = nn.GRU(input_size=indim, hidden_size=indim)
        self.linear = nn.Linear(finaldim, indim)
        self.ts = timestep
        self.indim = indim

    def forward(self, x):
        """Apply TSDecoder to layer input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        hidden = self.linear(x).unsqueeze(0)
        next = torch.zeros(1, len(x), self.indim).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        nexts = []
        for i in range(self.ts):
            next, hidden = self.gru(next, hidden)
            nexts.append(next.squeeze(0))
        return torch.cat(nexts, 1)
