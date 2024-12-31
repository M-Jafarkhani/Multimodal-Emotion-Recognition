"""Implements common unimodal encoders."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class Sequential(nn.Sequential):
    """Custom Sequential module for easier usage."""

    def __init__(self, *args, **kwargs):
        """Initialize Sequential Layer."""
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Apply args to Sequential Layer."""
        if "training" in kwargs:
            del kwargs["training"]
        return super().forward(*args, **kwargs)


class MLP(torch.nn.Module):
    """Two layered perceptron."""

    def __init__(
        self,
        indim,
        hiddim,
        outdim,
        dropout=False,
        dropoutp=0.1,
        output_each_layer=False,
    ):
        """Initialize two-layered perceptron.

        Args:
            indim (int): Input dimension
            hiddim (int): Hidden layer dimension
            outdim (int): Output layer dimension
            dropout (bool, optional): Whether to apply dropout or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            output_each_layer (bool, optional): Whether to return outputs of each layer as a list. Defaults to False.
        """
        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """Apply MLP to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        output = F.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        output2 = self.fc2(output)
        if self.dropout:
            output2 = self.dropout_layer(output)
        if self.output_each_layer:
            return [0, x, output, self.lklu(output2)]
        return output2


class GRU(torch.nn.Module):
    """Implements Gated Recurrent Unit (GRU)."""

    def __init__(
        self,
        indim,
        hiddim,
        dropout=False,
        dropoutp=0.1,
        flatten=False,
        has_padding=False,
        last_only=False,
        batch_first=True,
    ):
        """Initialize GRU Module.

        Args:
            indim (int): Input dimension
            hiddim (int): Hidden dimension
            dropout (bool, optional): Whether to apply dropout layer or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            flatten (bool, optional): Whether to flatten output before returning. Defaults to False.
            has_padding (bool, optional): Whether the input has padding or not. Defaults to False.
            last_only (bool, optional): Whether to return only the last output of the GRU. Defaults to False.
            batch_first (bool, optional): Whether to batch before applying or not. Defaults to True.
        """
        super(GRU, self).__init__()
        self.gru = nn.GRU(indim, hiddim, batch_first=True)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.flatten = flatten
        self.has_padding = has_padding
        self.last_only = last_only
        self.batch_first = batch_first

    def forward(self, x):
        """Apply GRU to input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=self.batch_first, enforce_sorted=False
            )
            out = self.gru(x)[1][-1]
        elif self.last_only:
            out = self.gru(x)[1][0]

            return out
        else:
            out, l = self.gru(x)
        if self.dropout:
            out = self.dropout_layer(out)
        if self.flatten:
            out = torch.flatten(out, 1)

        return out


class GRUWithLinear(torch.nn.Module):
    """Implements a GRU with Linear Post-Processing."""

    def __init__(
        self,
        indim,
        hiddim,
        outdim,
        dropout=False,
        dropoutp=0.1,
        flatten=False,
        has_padding=False,
        output_each_layer=False,
        batch_first=False,
    ):
        """Initialize GRUWithLinear Module.

        Args:
            indim (int): Input Dimension
            hiddim (int): Hidden Dimension
            outdim (int): Output Dimension
            dropout (bool, optional): Whether to apply dropout or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            flatten (bool, optional): Whether to flatten output before returning. Defaults to False.
            has_padding (bool, optional): Whether input has padding. Defaults to False.
            output_each_layer (bool, optional): Whether to return the output of every intermediate layer. Defaults to False.
            batch_first (bool, optional): Whether to apply batching before GRU. Defaults to False.
        """
        super(GRUWithLinear, self).__init__()
        self.gru = nn.GRU(indim, hiddim, batch_first=batch_first)
        self.linear = nn.Linear(hiddim, outdim)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.flatten = flatten
        self.has_padding = has_padding
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """Apply GRUWithLinear to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=True, enforce_sorted=False
            )
            hidden = self.gru(x)[1][-1]
        else:
            hidden = self.gru(x)[0]
        if self.dropout:
            hidden = self.dropout_layer(hidden)
        out = self.linear(hidden)
        if self.flatten:
            out = torch.flatten(out, 1)
        if self.output_each_layer:
            return [
                0,
                torch.flatten(x, 1),
                torch.flatten(hidden, 1),
                self.lklu(out),
            ]
        return out


class Identity(nn.Module):
    """Identity Module."""

    def __init__(self):
        """Initialize Identity Module."""
        super().__init__()

    def forward(self, x):
        """Apply Identity to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return x


class Transformer(nn.Module):
    """Extends nn.Transformer."""

    def __init__(self, n_features, dim):
        """Initialize Transformer object.

        Args:
            n_features (int): Number of features in the input.
            dim (int): Dimension which to embed upon / Hidden dimension size.
        """
        super().__init__()
        self.embed_dim = dim
        self.conv = nn.Conv1d(
            n_features, self.embed_dim, kernel_size=1, padding=0, bias=False
        )
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=5)
        self.transformer = nn.TransformerEncoder(layer, num_layers=5)

    def forward(self, x):
        """Apply Transformer to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if type(x) is list:
            x = x[0]
        x = self.conv(x.permute([0, 2, 1]))
        x = x.permute([2, 0, 1])
        x = self.transformer(x)[-1]
        return x


class Sequential2(nn.Module):
    """Implements a simpler version of sequential that handles inputs with 2 arguments."""

    def __init__(self, a, b):
        """Instatiate Sequential2 object.

        Args:
            a (nn.Module): First module to sequence
            b (nn.Module): Second module
        """
        super(Sequential2, self).__init__()
        self.model = nn.Sequential(a, b)

    def forward(self, x):
        """Apply Sequential2 modules to layer input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return self.model(x)
