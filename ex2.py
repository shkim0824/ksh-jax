from ksh-jax.evaluation import scoring

import jax
import jax.numpy as jnp

def f(rng):
  rng, key = jax.random.split(rng, 2)
  x = jax.random.uniform(rng, (10, 10))
  y = jax.random.uniform(key, (10, 10))
  return scoring.evaluate_ace(x, y)
