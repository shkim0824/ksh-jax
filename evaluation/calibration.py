import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import numpy as np  


__all__ = [
    "evaluate_ece",
    "evaluate_dee",
    "temperature_scaling"
]

# Expected Calibration Error
def evaluate_ece(softmax, labels, n_bins: int = 15):
    """
    Inputs
        softmax (Array): Softmax values with shape [N, K]
        labels (Array): True labels with shape [N, ]
        n_bins (int): number of bins to evaluate ece
    
    Outpus
        ece (float): Expected Calibration Error
    """
    # predicted labels and its confidence
    pred_conf = jnp.max(softmax, axis=1)
    pred_labels = jnp.argmax(softmax, axis=1)

    # binning
    ticks = jnp.linspace(0, 1, n_bins+1)
    bin_lowers = ticks[:-1] # list of lower bound of each bin
    bin_uppers = ticks[1:] # list of upper bound of each bin

    # ECE across bins
    acc = (pred_labels == labels) # boolean array. 1. (=True) for correct prediction
    ece = jnp.zeros(()) # 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # get boolean array s.t. pred_conf is inside bin range
        in_bin = (pred_conf > bin_lower) * (pred_conf < bin_upper)
        prop_in_bin = in_bin.mean() # fraction of elements contained in given bin
        acc_in_bin = (acc*in_bin).mean() # acc of in_bin elements
        avg_conf_in_bin = (pred_conf*in_bin).mean() # conf of in_bin elements

        ece += prop_in_bin * jnp.abs(avg_conf_in_bin - acc_in_bin) # conf - acc
    return ece

def evaluate_dee(de, v):
    diff = np.subtract(v, de)

    # first index s.t. diff <= 0
    k = np.argmax(diff <= 0)

    # if k = 0, then y is less than x[0]
    if k == 0:
        return k

    # if x >= len(de), then y is greater than x[len(de)-1]
    if k >= len(de):
        k = len(de) - 1

    # linear interpolation
    t = (v - de[k-1]) / (de[k] - de[k-1])
    interpolated_value = (1 - t) * (k-1) + t * k

    return interpolated_value

# Temperature Scaling
def temperature_scaling(softmax, labels, n_targets):
    one_hot = jax.nn.one_hot(labels, n_targets)
    def cNLL(t): # Get cNLL for each t
        logits = jnp.log(softmax) / t
        scaled_softmax = jax.nn.softmax(logits)
        return jnp.mean(-jnp.sum(one_hot * jnp.log(softmax+1e-12), axis=-1)) # Cross entropy loss

    # t = 1. # Initial Temperature
    # for epoch in range(300):
    #     loss, grads = jax.value_and_grad(cNLL)(t)
    #     t -= 1.*(jnp.power(0.1, epoch % 100)) * grads # SGD to find optimal t

    t = minimize(cNLL, jnp.asarray([1.0,]), method='BFGS', tol=1e-3).x[0] # Find optimzal temperature using scipy minimize

    # calibrated ECE
    cECE = evaluate_ece(jax.nn.softmax(jnp.log(softmax)/t), labels)

    return t, cNLL(t), cECE