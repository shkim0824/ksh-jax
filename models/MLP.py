import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from typing import Any, Sequence

class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs, training:bool): # training flag determine usage of moving average or not
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
                x = nn.Dropout(rate=0.1, deterministic=not training)(x)
        return x
