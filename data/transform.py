import jax
import jax.numpy as jnp

__all__ = [
    "TransformChain",
    "ToTensorTransform",
    "ToTensorNormalizeTransform",
    "ResizeTransform",
    "RandomHFlipTransform",
    "RandomCropTransform"
]


class TransformChain(object):
    def __init__(self, transforms):
        """
        Apply transform sequentially
        """
        self.transforms = transforms

    def __call__(self, rng, image):
        for _t in self.transforms:
            image = _t(rng, image)
        return image

class ToTensorTransform(object):
    def __init__(self):
        """
        Make image values between 0. ~ 1.
        """

    def __call__(self, rng, image):
        return image / 255.
    
class ToTensorNormalizeTransform(object):
    def __init__(self):
        """
        Substract 0.5, divide 0.5; general normalization
        """
    
    def __call__(self, rng, image):
        return ((image/255.) - 0.5 ) / 0.5

class ResizeTransform(object):
    def __init__(self, size=224):
        """
        Make image resolution to fit (224, 224, 3) to use in ViT; pre-trained in ImageNet
        """
        self.size = size
    
    def __call__(self, rng, image):
        return jax.image.resize(
                image     = image,
                shape     = (self.size, self.size, image.shape[2]),
                method    = jax.image.ResizeMethod.LANCZOS3,
                antialias = True,
        )

class RandomHFlipTransform(object):
    def __init__(self, prob=0.5):
        """
        Flip the image horizontally with the given probability

        Inputs:
            prob (float): probability of the flip
        """
        self.prob=prob

    def __call__(self, rng, image):
        return jnp.where(
            condition = jax.random.bernoulli(rng, self.prob), # 1 for prob=self.prob
            x         = jnp.flip(image, axis=1), # Flip image
            y         = image, # Or not
        )

class RandomCropTransform(object):

    def __init__(self, size, padding):
        """
        Crop the image at a random location with given size and padding.
        Inputs:
            size (int): desired output size of the crop.
            padding (int): padding on each border of the image before cropping.
        """
        self.size = size
        self.padding = padding

    def __call__(self, rng, image):
        # Add padding
        image = jnp.pad(
            array           = image,
            pad_width       = ((self.padding, self.padding),
                               (self.padding, self.padding),
                               (           0,            0),), # No pad for RGB channel
            mode            = 'constant',
            constant_values = 0,
        )

        # Random cropping position
        rng1, rng2 = jax.random.split(rng, 2)
        h0 = jax.random.randint(rng1, shape=(1,), minval=0, maxval=2*self.padding+1)[0] # output of randint is [x]. We get item of x
        w0 = jax.random.randint(rng2, shape=(1,), minval=0, maxval=2*self.padding+1)[0]

        # Slice image
        image = jax.lax.dynamic_slice(
            operand       = image,
            start_indices = (h0, w0, 0), # We do not crop rgb channel
            slice_sizes   = (self.size, self.size, image.shape[2]), # We do not crop rgb channel
        )

        return image
