"""
Implementation of the Terminator module. Notation for readability is provided
below. These names correspond directly to Figure 3 in the Zhang, 2024 paper.

sfne: slow-fast neural encoder
h: hyper
hk: hyperkernel
hzzw: hyperZZW
cm: channel mixer
ci: channel interaction
hi: hyper interaction
rgu: recursive gated unit
gb: group-based
bstan: batch standardization
istan: instance standardization
si_glu: sigmoid gated linear unit
"""

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def istan(x):
    pass

def bstan(x):
    pass

def si_glu(x):
    pass

def gelu(x):
    return nn.functional.gelu(x)

class GlobalHyperZZW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class LocalHyperZZW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

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

class MutualHyperKernelGeneration(nn.Module):
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
    def __init__(self, lamb: float):
        super().__init__()
        self.slow_hzzw = GlobalHyperZZW()
        self.hzzw_rgu_cm = RGUChannelMixer()
        self.rgu_cm = RGUChannelMixer()
        self.hci = HyperChannelInteraction()
        self.cm = ChannelMixer()
        self.hzzw5 = LocalHyperZZW()
        self.hzzw7 = LocalHyperZZW()
        self.hzzw9 = LocalHyperZZW()
        self.muhkgen = MutualHyperKernelGeneration() 
        self.hi = HyperInteraction()

        self.cb_layer = ChannelBottleneckLayer(lamb)
        self.gb_ibs = GroupbasedInstanceBatchStandardization() 
        self.dropout = nn.Dropout()

    def forward(self, inp):
        # Context-dependent hyperkernel
        context_dep_hk = self.slow_hyperzzw(inp)

        # RGU Channel Mixer Steps
        hk_rgu_cm_out = self.hzzw_rgu_cm(inp * context_dep_hk)
        rgu_cm_out = self.rgu_cm(inp)
        rgu_out = self.istan(hk_rgu_cm_out + rgu_cm_out)

        # Hyper Channel Interaction Step
        hci_out = self.istan(self.hci(inp))

        # Channel Mixer Steps
        cm_out = self.istan(self.cm(inp))

        # Mutal Hyperkernel Generation Step
        mu_hk = self.muhkgen(inp)

        # Hyper Interaction Step
        # TODO: This seems wrong becaues the hk comes post-is shenanigans
        hi_out = self.hi(context_dep_hk, inp)

        # Calculating channels 1 through 9 (left to right in Figure 3)
        c1 = rgu_out
        c3 = context_dep_hk * rgu_out
        c2 = self.si_glu(c3) 
        c4 = context_dep_hk * cm_out
        c5 = context_dep_hk * hci_out
        c6 = nn.function.gelu(self.bstan(self.hzzw5(mu_hk, hci_out)))
        c7 = nn.function.gelu(self.bstan(self.hzzw7(mu_hk, cm_out)))
        c8 = nn.function.gelu(self.bstan(self.hzzw9(mu_hk, cm_out))) # this is weird because the last UML is flipped on operations
        c9 = self.hi(inp, c3)
    
        out = self.channel_bottleneck_layer(out)
        out = self.g_ibs(out) # needs checking I think ibs gets swapped
        out = nn.functional.gelu(out)
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