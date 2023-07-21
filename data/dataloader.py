import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial

from data.transform import *


def _buildLoader(images, labels, batch_size, steps_per_epoch, rng=None, shuffle=False, transform=None):
    # Shuffle Indices
    indices = jax.random.permutation(rng, len(images)) if shuffle else jnp.arange(len(images)) # Make shuffled indices
    indices = indices[:steps_per_epoch*batch_size] # Batch size may not be divisor of length of images. We drop left ones.
    indices = indices.reshape((steps_per_epoch, batch_size,))
    for batch_idx in indices:
        batch = {'images': jnp.array(images[batch_idx]), 'labels': jnp.array(labels[batch_idx])}
        if transform is not None:
            if rng is not None:
                _, rng = jax.random.split(rng)
            sub_rng = None if rng is None else jax.random.split(rng, batch['images'].shape[0])
            batch['images'] = transform(sub_rng, batch['images'])
        yield batch

def buildLoader(train_set, test_set, rng, batch_size, test_size=None):
    """
    Inputs
        train_set: PyTorch datasets to train
        test_set: PyTorch datasets to test
        rng (PRNGKey): Initial PRNGKey
        batch_size: hyper parameter
        test_size: hyper parameter; used to test. It can be much larger than batch_size, but should consider memory issue
    
    Returns
        dictionary of trn_loader, vl_loader, tst_loader
    """

    # Make validation set
    train_images, val_images = train_test_split(train_set.data, test_size=5000, random_state=42)
    train_labels, val_labels = train_test_split(train_set.targets, test_size=5000, random_state=42)

    # We use img converted to jnp dtype
    train_images = np.array(train_images, dtype=jnp.float32)
    val_images = np.array(val_images, dtype=jnp.float32)
    test_images = np.array(test_set.data, dtype=jnp.float32)

    # Make labels to numpy array
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_set.targets)

    # transform for CIFAR-10
    transform = TransformChain([RandomHFlipTransform(0.5),
                                RandomCropTransform(size=32, padding=4),
                                ToTensorTransform()])

    # Naive data loader. We should put rng later in real-time usage
    if test_size is None:
        test_size = len(test_images)
    trn_steps_per_epoch = len(train_images) // batch_size
    tst_steps_per_epoch = len(test_images) // test_size
    vl_steps_per_epoch = len(val_images) // test_size

    trn_loader = partial(_buildLoader,
                            images=train_images,
                            labels=train_labels,
                            batch_size=batch_size,
                            steps_per_epoch=trn_steps_per_epoch,
                            shuffle=True,
                            transform=jit(vmap(transform)))

    vl_loader = partial(_buildLoader,
                        images=val_images,
                        labels=val_labels,
                        batch_size=test_size,
                        steps_per_epoch=vl_steps_per_epoch,
                        shuffle=False,
                        transform=jit(vmap(ToTensorTransform())))

    tst_loader = partial(_buildLoader,
                        images=test_images,
                        labels=test_labels,
                        batch_size=test_size,
                        steps_per_epoch=tst_steps_per_epoch,
                        shuffle=False,
                        transform=jit(vmap(ToTensorTransform())))
    return {"trn_loader": trn_loader, "vl_loader": vl_loader, "tst_loader": tst_loader}






