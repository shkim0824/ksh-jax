import jax
import jax.numpy as jnp
import flax
from flax import linen as nn

__all__ = [
    "Linear",
    "Linear_Dropout",
    "Linear_BatchEnsemble"
]


def normalp1_init():
    returned_function = nn.initializers.normal()

    def add_1(*args, **kwargs):
        result = returned_function(*args, **kwargs)
        return result + 1.

    return add_1


class Linear(nn.Module):
    features: int
    use_bias: bool = True
    w_init: Callable = jax.nn.initializers.he_normal()
    b_init: Callable = jax.nn.initializers.normal()

    @nn.compact
    def __call__(self, x, **kwargs):
        """
        Inputs:
            x (NumPy Array): Input array with shape [N, C1, ]

        Returns:
            y (NumPy Array): Output array with shape [N, C2, ]
        """
        w = jnp.asarray(self.param('w', self.w_init, (x.shape[-1], self.features,)), x.dtype) # self.param(name, init_fn, **kargs)
        y = jnp.dot(x, w)
        if self.use_bias:
            b = jnp.asarray(self.param('b', self.b_init, (self.features,)), x.dtype)
            y = jnp.add(y, jnp.reshape(b, (1, -1,))) # b is added as mini batch
        return y

class Linear_Dropout(nn.Module):
    features: int
    use_bias: bool = True
    drop_rate: float = 0.5
    deterministic: Optional[bool] = False
    w_init: Callable = jax.nn.initializers.he_normal()
    b_init: Callable = jax.nn.initializers.normal()

    @nn.compact
    def __call__(self, x, **kwargs):
        """
        Inputs:
            x (NumPy Array): Input array with shape [N, C1, ]

        Returns:
            y (NumPy Array): Output array with shape [N, C2, ]
        """
        deterministic = kwargs.pop('deterministic', True)
        # flax merge_param allows module hyperparameter can be passed to __call__
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)

        if not deterministic:
            rng = self.make_rng('dropout')
            keep = 1.0 - self.drop_rate
            mask = jax.random.bernoulli(rng, p=keep, shape=x.shape)
            x = jax.lax.select(mask, x / keep, jnp.zeros_like(x)) # inverted dropout

        w = jnp.asarray(self.param('w', self.w_init, (x.shape[-1], self.features,)), x.dtype) # self.param(name, init_fn, **kargs)
        y = jnp.dot(x, w)
        if self.use_bias:
            b = jnp.asarray(self.param('b', self.b_init, (self.features,)), x.dtype)
            y = jnp.add(y, jnp.reshape(b, (1, -1,))) # b is added as mini batch
        return y

class Linear_BatchEnsemble(nn.Module):
    ensemble_size: int
    features: int
    use_bias: bool = True
    w_init: Callable = jax.nn.initializers.he_normal()
    b_init: Callable = jax.nn.initializers.normal()
    r_init: Callable = normalp1_init()
    s_init: Callable = normalp1_init()

    @nn.compact
    def __call__(self, x, **kwargs):
        """
        Inputs:
            x (NumPy Array): Input array with shape [N, C1, ]

        Returns:
            y (NumPy Array): Output array with shape [N, C2, ]
        """
        # X \odot R
        # R is initialized vectorizely, rows corresponds to each ensemble member
        r = jnp.asarray(self.param('r', self.r_init, (self.ensemble_size, x.shape[-1], )), x.dtype)
        # X is reshaped to do Hadamard Product, (ensemble, minibatch, shape)
        # We divide mini batch into ensemble size
        x = jnp.reshape(x, (self.ensemble_size, x.shape[0] // self.ensemble_size, -1,))
        # R is reshaped to do Hadamard Product, we do broadcast along minibatch dimension
        x = jnp.multiply(x, jnp.reshape(r, (self.ensemble_size, 1, -1,)))

        # (X \odot R) W
        w = jnp.asarray(self.param('w', self.w_init, (x.shape[-1], self.features)), x.dtype)
        # dot product XW is conducted at dim 2; dimension of data
        y = jax.lax.dot_general(x, w, (((2,), (0,),), ((), (),)))

        if self.use_bias:
            b = jnp.asarray(self.param('b', self.b_init, (self.features,)), x.dtype)
            # Also bias is shared along ensemble and minibatch
            y = jnp.add(y, jnp.reshape(b, (1, 1, -1,)))

        # ((X \odot R) W) \odot S
        # S is initialized vectorizely, rows corresponds to each ensemble member
        s = jnp.asarray(self.param('s', self.s_init, (self.ensemble_size, y.shape[-1],)), x.dtype)
        # S is reshaped to do Hadamard Product, we do broadcast along minibatch dimension
        y = jnp.multiply(y, jnp.reshape(s, (self.ensemble_size, 1, -1,)))

        y = jnp.reshape(y, (y.shape[0] * y.shape[1], -1,))

        return y
