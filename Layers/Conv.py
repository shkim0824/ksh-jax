import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import torch

__all__ = [
    "Conv",
    "Conv_BatchEnsemble"
]

def normalp1_init():
    returned_function = nn.initializers.normal()

    def add_1(*args, **kwargs):
        result = returned_function(*args, **kwargs)
        return result + 1.

    return add_1

class Conv(nn.Module):
    channels: int
    kernel_size: int
    stride: int = 1
    padding: Union[str, int] = 'SAME'
    use_bias: bool = True
    w_init: Callable = jax.nn.initializers.he_normal()
    b_init: Callable = jax.nn.initializers.normal()

    @nn.compact
    def __call__(self, x, **kwargs):
        """
        Inputs:
            x (Array): An input array with shape [N, H1, W1, C1,].

        Returns:
            y (Array): An output array with shape [N, H2, W2, C2,].
        """
        in_channels = x.shape[-1]
        w_shape = (self.kernel_size, self.kernel_size, in_channels, self.channels,)
        padding = self.padding if isinstance(self.padding, str) else [
            (self.padding, self.padding), (self.padding, self.padding),
        ]

        w = jnp.asarray(self.param('w', self.w_init, w_shape), x.dtype)
        # Conv operation
        y = jax.lax.conv_general_dilated(
            x, w, (self.stride, self.stride,), padding,
            lhs_dilation=(1, 1,), rhs_dilation=(1, 1,),
            dimension_numbers = jax.lax.ConvDimensionNumbers((0,3,1,2,), (3,2,0,1,), (0,3,1,2,)),
        )
        if self.use_bias:
            b = jnp.asarray(self.param('b', self.b_init, (self.channels,)), x.dtype)
            y = jnp.add(y, jnp.reshape(b, (1, 1, 1, -1,))) # Bias is same for mini bacth, each features

        return y

class Conv_BatchEnsemble(nn.Module):
    ensemble_size: int
    channels: int
    kernel_size: int
    stride: int = 1
    padding: Union[str, int] = 'SAME'
    use_bias: bool = True
    w_init: Callable = jax.nn.initializers.he_normal()
    b_init: Callable = jax.nn.initializers.normal()
    r_init: Callable = normalp1_init()
    s_init: Callable = normalp1_init()

    @nn.compact
    def __call__(self, x, **kwargs):
        """
        Inputs:
            x (Array): An input array with shape [N, H1, W1, C1,].

        Returns:
            y (Array): An output array with shape [N, H2, W2, C2,].
        """

        in_channels = x.shape[-1]
        w_shape = (self.kernel_size, self.kernel_size, in_channels, self.channels,)
        padding = self.padding if isinstance(self.padding, str) else [
            (self.padding, self.padding), (self.padding, self.padding),
        ]

        # X \odot R
        # R is initialized vectorizely, rows corresponds to each ensemble member
        # Note that we use same r for each channel; i.e., height-width share r
        r = jnp.asarray(self.param('r', self.r_init, (self.ensemble_size, x.shape[-1], )), x.dtype)
        # X is reshaped to do Hadamard Product, (ensemble, minibatch, shape)
        x = jnp.reshape(x, (self.ensemble_size, x.shape[0] // self.ensemble_size, x.shape[1], x.shape[2], -1))
        # R is reshaped to do Hadamard Product
        # Since r is shared for each channel,
        # we do broadcast not only along minibatch dimension but also along height-width
        x = jnp.multiply(x, jnp.reshape(r, (self.ensemble_size, 1, 1, 1, -1,)))

        # (X \odot R) W
        w = jnp.asarray(self.param('w', self.w_init, w_shape), x.dtype)
        # Conv operation
        y = jax.lax.conv_general_dilated(
          # x is reshaped to do conv operation
          jnp.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], -1)),
            w, (self.stride, self.stride,), padding,
            lhs_dilation=(1, 1,), rhs_dilation=(1, 1,),
            dimension_numbers = jax.lax.ConvDimensionNumbers((0,3,1,2,), (3,2,0,1,), (0,3,1,2,)),
        )

        if self.use_bias:
            b = jnp.asarray(self.param('b', self.b_init, (self.channels,)), x.dtype)
            # Also bias is shared along ensemble and height-width
            y = jnp.add(y, jnp.reshape(b, (1, 1, 1, -1,)))

        # ((X \odot R) W) \odot S
        # S is initialized vectorizely, rows corresponds to each ensemble member
        # Note that we use same s for each channel; i.e., height-width share s
        s = jnp.asarray(self.param('s', self.s_init, (self.ensemble_size, y.shape[-1],)), x.dtype)
        # Y is reshaped to do Hadamard Product, since we use shared s for each channel
        y = jnp.reshape(y, (self.ensemble_size, y.shape[0] // self.ensemble_size, y.shape[1], y.shape[2], -1))
        y = jnp.multiply(y, jnp.reshape(s, (self.ensemble_size, 1, 1, 1, -1,)))
        # Get back y into batch form
        y = jnp.reshape(y, (y.shape[0] * y.shape[1], y.shape[2], y.shape[3], -1))
        return y
