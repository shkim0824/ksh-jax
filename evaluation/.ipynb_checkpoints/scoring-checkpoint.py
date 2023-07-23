import jax
import jax.numpy as jnp

__all__ = [
    "evaluate_ce",
    "evaluate_acc",
    "evaluate_bs"
]

# Cross Entropy loss
def evaluate_ce(softmax, one_hot):
    return jnp.mean(-jnp.sum(one_hot * jnp.log(softmax+1e-12), axis=-1))

# Test Accuracy
def evaluate_acc(logits, labels):
    return jnp.mean(jnp.argmax(logits, axis=1) == labels)

# Brier Score
def evaluate_bs(softmax, one_hot):
    return jnp.mean(jnp.power(one_hot - softmax, 2))
