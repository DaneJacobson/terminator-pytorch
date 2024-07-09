"""
Implementation of the Terminator model based on the HyperZZW operator presented
in Zhang 2024: https://arxiv.org/pdf/2401.17948. Module names correspond 
directly to the paper's for readability.
"""

import torch
from torch import nn

import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# -------------------------------- #
# -- Functional transformations -- #
# -------------------------------- #

def istan(x):
    pass

def bstan(x):
    pass

def gelu(x):
    pass

def si_glu(x):
    pass

def normalize(out):
    """Normalizes hidden ouputs of slow net"""
    return nn.functional.normalize(out.view(out.shape[0], -1)).view_as(out)

# -------------------- #
# -- Custom Modules -- #
# -------------------- #

class MAGNetLayer(torch.nn.Module):
    """
    Gabor-like filter as used in GaborNet for hyper kernel. Used to generate
    hyperkenels for some HyperZZW operators.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_channels: int,
        omega_0: float,
        alpha: float,
        beta: float,
    ):
        super().__init__()

        # Define type of linear to use
        # Linear = getattr(linear, f"Linear{data_dim}d")

        self.gamma = torch.distributions.gamma.Gamma(alpha, beta).sample(
                (hidden_channels, data_dim)
            )
        
        self.linear = nn.Linear(data_dim, hidden_channels, bias=True)
        self.linear.weight.data *= (
            2 * np.pi * omega_0 * self.gamma.view(*self.gamma.shape, *((1,) * data_dim))
        )
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        return torch.sin(self.linear(x))


class GlobalHyperZZW(nn.Module):
    """
    Global HyperZZW K=N module, using element-wise multiplication

    Generated using coordinate-based implicit MLPs a.k.a. multiplicative filter
    networks (MFNs) from Fathony et al. 2020: https://openreview.net/pdf?id=OmtmcPkkhT
    """ 
    def __init__(
        self, 
        data_dim, 
        hidden_channels, 
        num_layers, 
        out_channels, 
        kernel_size,
        omega_0: float = 2000.0,
        alpha: float = 6.0,
        beta: float = 1.0,
    ):
        super().__init__() 
        
        self.hidden_channels = hidden_channels
        
        # Define type of linear to use
        # Linear = getattr(linear, f"Linear{data_dim}d")
        
        # Hidden layers
        self.linears = nn.ModuleList(
            [
                nn.Linear(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        
        # Final layer
        self.output_linear = nn.Linear(
            in_channels=hidden_channels,
            out_channels=out_channels,
            bias=True,
        )
        
        self.filters = nn.ModuleList(
            [
                MAGNetLayer(
                    data_dim=data_dim,   
                    hidden_channels=hidden_channels,
                    omega_0=omega_0,
                    alpha=alpha / (layer + 1),
                    beta=beta,
                )
                for layer in range(num_layers + 1)
            ]
        )
        
        # Initialize
        for lin in self.linears:    
            torch.nn.init.kaiming_uniform_(
                tensor=lin.weight,
                nonlinearity="linear"
            )
            if lin.bias is not None:
                lin.bias.data.fill_(1.0)
        torch.nn.init.kaiming_uniform_(
            tensor=self.output_linear.weight,
            nonlinearity="linear"
        )
        self.output_linear.bias.data.fill_(0.0)
        self.bias_p = torch.nn.Parameter(
            torch.zeros(1, 1, kernel_size, kernel_size), requires_grad=True
        )
        
    def forward(self, coords, x):
        out = self.filters[0](coords)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linears[i - 1](out) + self.bias_p.repeat(1, self.hidden_channels, 1, 1)
        out = self.output_linear(out)
        return out

class GlobalChannelHyperZZW(torch.nn.Module):
    """
    The slow net to generate hyper-kernel for hyper-channel interaction.
    """
    def __init__(
        self,
        data_dim: int,
        kernel_size: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        omega_0: float = 2000.0,
        alpha: float = 6.0,
        beta: float = 1.0,
        **kwargs,
    ):
        # call super class
        super().__init__()
        
        self.hidden_channels = hidden_channels
        
        # Define type of linear to use
        # Linear = getattr(linear, f"Linear{data_dim}d")
        
        # Hidden layers
        self.linears = torch.nn.ModuleList(
            [
                nn.Linear(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        
        # Final layer
        self.output_linear = nn.Linear(
            in_channels=hidden_channels,
            out_channels=out_channels,
            bias=True,
        )
        
        self.filters = torch.nn.ModuleList(
            [
                MAGNetLayer(
                    data_dim=data_dim,   
                    hidden_channels=hidden_channels,
                    omega_0=omega_0,
                    alpha=alpha / (layer + 1),
                    beta=beta,
                )
                for layer in range(num_layers + 1)
            ]
        )
        
        # Initialize
        for idx, lin in enumerate(self.linears):
            torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity="linear")
            if lin.bias is not None:
                lin.bias.data.fill_(1.0)
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        self.output_linear.bias.data.fill_(0.0)
        
        # Bias
        self.bias_p = torch.nn.Parameter(
            torch.zeros(1, 1, kernel_size), requires_grad=True
        )
        
    def forward(self, coords, x):
        out = self.filters[0](coords)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linears[i - 1](out) + self.bias_p.repeat(1, self.hidden_channels, 1)
        out = self.output_linear(out)
        return out

class LocalHyperZZW(nn.Module):
    """
    Local HyperZZW operator, using sliding-window based convolution

    Generated using coordinate-based implicit MLPs a.k.a. multiplicative filter
    networks (MFNs) from Fathony et al. 2020: https://openreview.net/pdf?id=OmtmcPkkhT.
    
    Note that, for the "local" MFN, the HyperZZW operator is incorporated into 
    generation process for context-dependency.
    """
    def __init__(
        self,
        data_dim: int,
        kernel_size: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        omega_0: float = 2000.0,
        alpha: float = 6.0,
        beta: float = 1.0,
    ):
        # call super class
        super().__init__()
        
        self.hidden_channels = hidden_channels
        
        # Define type of linear to use
        # Linear = getattr(linear, f"Linear{data_dim}d")
        
        # Hidden layers
        self.linears = torch.nn.ModuleList(
            [
                nn.Linear(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        
        # Final layer
        self.output_linear = nn.Linear(
            in_channels=hidden_channels,
            out_channels=out_channels,
            bias=True,
        )
        
        self.filters = torch.nn.ModuleList(
            [
                MAGNetLayer(
                    data_dim=data_dim,   
                    hidden_channels=hidden_channels,
                    omega_0=omega_0,
                    alpha=alpha / (layer + 1),
                    beta=beta,
                )
                for layer in range(num_layers + 1)
            ]
        )
        
        # Initialize
        for idx, lin in enumerate(self.linears):
            torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity="linear")
            if lin.bias is not None:
                lin.bias.data.fill_(1.0)
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        self.output_linear.bias.data.fill_(0.0)
        
        # Bias
        self.bias_pk = torch.nn.Parameter(
            torch.zeros(1, 1, kernel_size, kernel_size), requires_grad=True
        )
        
        # Transform input to add context-dependency
        NonlinearType = getattr(torch.nn, 'GELU')
        self.data_dim = data_dim
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        
        # Define type of linear to use
        # Linear = getattr(linear, f"Linear{data_dim}d")
        
        self.fast_reduce = torch.nn.Conv2d(self.out_channels, self.hidden_channels, 1, 1, 0, bias=True)
        self.fast_fc1 = torch.nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=True)
        self.fast_gelu1 = NonlinearType()
        self.fast_linears_x = torch.nn.ModuleList(
            [
                nn.Linear(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        # Initialize
        for idx, lin in enumerate(self.fast_linears_x):
            torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity="linear")
            if lin.bias is not None:
                lin.bias.data.fill_(1.0)
                
        self.alphas = [nn.Parameter(torch.Tensor(hidden_channels).fill_(1)) for _ in range(num_layers)]
        self.betas = [nn.Parameter(torch.Tensor(hidden_channels).fill_(0.1)) for _ in range(num_layers)]
        
    def s_renormalization(self, out, alpha, beta):
        out = out.transpose(0, 1)
        
        delta = out.data.clone()
        assert delta.shape == out.shape

        v = (-1,) + (1,) * (out.dim() - 1)
        out_t = alpha.view(*v) * delta + beta.view(*v) * normalize(out)
        
        return out_t.transpose(0, 1)
    
    def forward(self, coords, x):
        x_gap = torch.nn.functional.adaptive_avg_pool2d(
            x,
            (self.kernel_size,) * self.data_dim,
        )
        x_gap = self.fast_reduce(x_gap)
        x_gap = x_gap.mean(axis=0, keepdims=True)
        x_gap = self.fast_fc1(x_gap)
        x_gap = self.fast_gelu1(x_gap)
        
        out = self.filters[0](coords)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linears[i - 1](out) + self.bias_pk.repeat(1, self.hidden_channels, 1, 1)
            out = self.s_renormalization(out, self.alphas[i-1].cuda(), self.betas[i-1].cuda())
            out = out * self.fast_linears_x[i - 1](x_gap)

        out = self.output_linear(out)
        
        return out

class RGUChannelMixer(nn.Module):
    """
    RGUChannelMixer, which is recursion applied to a gated linear unit (GLU).
    """
    def __init__(
        self,
        in_channels: int,
        bias_size: int,
    ):
        super().__init__()
        self.hidden_channels = in_channels
        self.bias_size = bias_size

        # Declare modules
        self.WK = nn.Linear(in_channels, in_channels)
        self.WV = nn.Linear(in_channels, in_channels)
        self.WQ = nn.Linear(in_channels, in_channels)
        self.WY = nn.Linear(in_channels, in_channels)
        self.GeLU = nn.functional.gelu
        self.IS = nn.InstanceNorm2d(num_features=in_channels, momentum=1.0,) # TODO: This feels wrong I think it should be the default 0.1
        self.bias_p = torch.nn.Parameter(torch.zeros(1, 1, bias_size, bias_size), requires_grad=True) # TODO: Simplify

        # Initialize parameters
        nn.init.kaiming_uniform_(self.WK.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.WV.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.WQ.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.WY.weight, nonlinearity="linear")
        nn.init.constant_(self.WK.bias, 0.0)
        nn.init.constant_(self.WV.bias, 0.0)
        nn.init.constant_(self.WQ.bias, 0.0)
        nn.init.constant_(self.WY.bias, 0.0)

    def forward(self, x):
        K = self.WK(x)
        V = self.WV(K)
        Q = self.GeLU(self.IS(K*self.WQ(V))) + self.bias_p.repeat(x.shape[0], self.hidden_channels, 1, 1)
        Y = self.WY(Q)
        return Y

class ChannelMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class RGUChannelMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class ChannelBottleneckLayer(nn.Module):
    def __init__(self, lamb: float):
        super().__init__()
        self.lamb = lamb

    def forward(self, x):
        pass

class GroupbasedInstanceBatchStandardization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass 

class HyperChannelInteraction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class HyperInteraction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class SFNE(nn.Module):
    """
    Implementation of the Slow-Fast Neural Encoding (SFNE) block presented in
    Figure 3 of Zhang 2024: https://arxiv.org/pdf/2401.17948.
    """
    def __init__(self, bottleneck_lambda: int):
        super().__init__()
        self.global_hyperzzw = GlobalHyperZZW()
        self.slow_rgu_mlp = RGUChannelMixer()
        self.fast_rgu_mlp = RGUChannelMixer()
        self.hyper_channel_interaction = HyperChannelInteraction()
        self.channel_mixer = ChannelMixer()
        self.hyperzzw5 = LocalHyperZZW(5)
        self.hyperzzw7 = LocalHyperZZW(7)
        self.hyperzzw9 = LocalHyperZZW(9)
        self.hyper_interaction = HyperInteraction()
        self.channel_bottleneck_layer = ChannelBottleneckLayer(bottleneck_lambda)
        self.g_ibs = GroupbasedInstanceBatchStandardization() 
        self.dropout = nn.Dropout()

    def forward(self, inp):
        # Generate context-dependent global hyperkernel
        global_hyperkernel = self.global_hyperzzw(inp)

        # RGUChannelMixers
        slow_rgu_mlp_out = self.slow_rgu_mlp(global_hyperkernel)
        fast_rgu_mlp_out = self.fast_rgu_mlp(inp)
        fast_out = istan(slow_rgu_mlp_out + fast_rgu_mlp_out)

        # HyperChannelInteraction
        hci_out = istan(self.hyper_channel_interaction(inp))

        # ChannelMixer
        cm_out = istan(self.channel_mixer(inp))

        # Local HyperZZWs
        # TODO: This part is tunable, need to prop drill a config
        hyperzzw5_out = self.hyperzzw5(inp, hci_out)
        hyperzzw7_out = self.hyperzzw7(inp, cm_out)
        hyperzzw9_out = self.hyperzzw9(inp, cm_out)

        # HyperInteraction
        hi_out = self.hyper_interaction(inp, global_hyperkernel)

        # Process channels 1 through 9 (left to right in Figure 3 of paper)
        c1 = fast_out
        c2 = si_glu(global_hyperkernel * fast_out) 
        c3 = global_hyperkernel * fast_out
        c4 = global_hyperkernel * cm_out
        c5 = global_hyperkernel * hci_out
        c6 = gelu(bstan(hyperzzw5_out)) # TODO: This part is tunable, need to prop drill a config
        c7 = gelu(bstan(hyperzzw7_out))
        c8 = gelu(bstan(hyperzzw9_out))
        c9 = hi_out

        # Post Channel Concatenation Steps 
        out = self.channel_bottleneck_layer([c1, c2, c3, c4, c5, c6, c7, c8, c9])
        out = self.g_ibs(out) # needs checking I think ibs gets swapped
        out = gelu(out)
        out = self.dropout(out)
        return out

class Terminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.sfne_blocks = nn.Sequential([self.SFNE() for _ in range(3)])

    def forward(self, x):
        x = self.sfne_blocks(x)
        x = nn.AvgPool2d(x)
        return x