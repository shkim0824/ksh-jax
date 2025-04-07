import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

class IdentityShortcut(nn.Module):
    channels: int
    strides: int
    expansion: int = 1

    @nn.compact
    def __call__(self, x):
        pad_offset = self.expansion * self.channels - x.shape[-1] # x.shape[-1] is original channel dimension
        return jnp.pad(
            array           = x[:, ::self.strides, ::self.strides, :], # get reduced shape of x
            pad_width       = ((0,0), (0,0), (0,0), (0, pad_offset)), # Add zero padding to channel
            mode            = 'constant',
            constant_values  = 0, # zero padding
        )

class ResidualBlock(nn.Module):
    channels: int
    strides: int = 1
    shortcut: nn.Module = IdentityShortcut # identity or projection

    @nn.compact
    def __call__(self, x, training=False):
        y = nn.Conv(features = self.channels, kernel_size=(3,3), strides=self.strides)(x)
        y = nn.BatchNorm(use_running_average=not training)(y)
        y = nn.relu(y)
        y = nn.Conv(features = self.channels, kernel_size=(3,3))(y)
        y = nn.BatchNorm(use_running_average=not training)(y)

        if self.strides != 1 or x.shape[-1] != self.channels * 1: # We have to modify x to match shape
            y = y + self.shortcut(channels  = self.channels,
                                  strides    = self.strides,
                                  expansion = 1)(x)
        else:
            y = y + x

        y = nn.relu(y)

        return y

class ResNet32(nn.Module):
    """ ResNet-32 Structure for CIFAR-10
    Starting Block: 3 by 3 conv, 16
    Residual Block: 16 channel * n - 32 channel * n - 64 channel * n, n=5 for ResNet-32
    """

    @nn.compact
    def __call__(self, x, training=False):
        # Starting Block: 3 by 3 conv
        x = nn.Conv(features = 16, kernel_size=(3,3))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        # Intermediate Residual Blocks
        for i in range(5):
            x = ResidualBlock(channels=16)(x, training)
            # jax.debug.print("{}", x.shape)

        for channel in [32, 64]:
            x = ResidualBlock(channels=channel, strides=2)(x, training)
            for i in range(4):
                x = ResidualBlock(channels=channel)(x, training)

        # Global Average Pooling
        x = jnp.mean(x, [1, 2])

        # Classifier
        x = nn.Dense(features=10)(x) # Classifier, return logits

        return x

def create_res32(rng, lr_fn):
    model = ResNet32()
    variables = model.init(rng, jnp.ones((4, 32, 32, 3)), training=False)
    print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(variables)))

    class TrainState(train_state.TrainState): # We don't use dropout in ResNet
        batch_stats: Any

    state = TrainState.create(
        apply_fn = model.apply,
        params = variables['params'],
        batch_stats = variables['batch_stats'], # batch state
        tx = optax.sgd(learning_rate=lr_fn, momentum=0.9, nesterov=True) # learning_rate_fn automatically determine learning rate
    )
    return state
