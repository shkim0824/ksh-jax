import jax
import jax.numpy as jnp
import dm_pix

__all__ = [
    "TransformChain",
    "ToTensorTransform",
    "NormalizeTransform",
    "ResizeTransform",
    "RandomHFlipTransform",
    "RandomCropTransform",
    "RandomCropScaleTransform",
    "CutoutTransform",
    "BrightnessTransform",
    "ColorJitterTransform"
]


class TransformChain(object):
    def __init__(self, transforms):
        """
        Apply transform sequentially
        """
        self.transforms = transforms

    def __call__(self, rng, image):
        for _t in self.transforms:
            rng, key = jax.random.split(rng, 2)
            image = _t(key, image)
        return image

class ToTensorTransform(object):
    def __init__(self):
        """
        Make image values between 0. ~ 1.
        """

    def __call__(self, rng, image):
        return image / 255.

class NormalizeTransform(object):
    def __init__(self):
        """
        Substract 0.5, divide 0.5; general normalization
        """
    
    def __call__(self, rng, image):
        return (image - 0.5 ) / 0.5

class ResizeTransform(object):
    def __init__(self, size=224):
        """
        Make image resolution to fit (224, 224, 3) to use in ViT; pre-trained in ImageNet
        
        Inputs:
            size (int): desired size of output image
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

class RandomCropScaleTransform(object):

    def __init__(self, size, scale=0.9):
        """
        Crop the image at a random location with given size and padding.
        
        Inputs:
            size (int): desired output size of the crop.
            scale (float): desired scaling of input. float between 0.08 and 1.0
        """
        self.size = size
        self.scale = scale

    def __call__(self, rng, image):
        # Random cropping position
        h, w = image.shape[0], image.shape[1] # original resolution
        new_h = int(h * self.scale)
        new_w = int(w * self.scale)
        
        rng1, rng2 = jax.random.split(rng, 2)
        h0 = jax.random.randint(rng1, shape=(1,), minval=0, maxval=h-new_h+1)[0] # output of randint is [x]. We get item of x
        w0 = jax.random.randint(rng2, shape=(1,), minval=0, maxval=w-new_w+1)[0]

        # Slice image
        new_image = jax.lax.dynamic_slice(
            operand       = image,
            start_indices = (h0, w0, 0), # We do not crop rgb channel
            slice_sizes   = (new_h, new_w, image.shape[2]), # We do not crop rgb channel
        )
        
        return jax.image.resize(
                image     = new_image,
                shape     = (self.size, self.size, image.shape[2]),
                method    = jax.image.ResizeMethod.LINEAR,
                antialias = True,
        )
class CutoutTransform(object):
    def __init__(self, patch_size, prob=0.5):
        """
        Cutout the image of given patch size
        
        Inputs
            patch_size (int): cut out patch isze
            prob (float): probability of cut out
        """
        self.patch_size = patch_size
        self.prob = prob
    
    def __call__(self, rng, image):
        h, w, c = image.shape[0], image.shape[1], image.shape[2] # original resolution
        mask = jnp.ones((h, w, c), dtype=jnp.float32)
        
        # random coordinates to cutout
        keys = jax.random.split(rng, 3)
        h0 = jax.random.randint(keys[0], shape=(1,), minval=0, maxval=h-self.patch_size)[0] # output of randint is [x]. We get item of x
        w0 = jax.random.randint(keys[1], shape=(1,), minval=0, maxval=w-self.patch_size)[0]
        
        # cutout image
        mask = jax.lax.dynamic_update_slice(mask, jnp.zeros((self.patch_size, self.patch_size, c), dtype=jnp.float32), (h0, w0, c))
        img_cutout = image * mask
        
        return jnp.where(jax.random.bernoulli(keys[2], self.prob),
                         img_cutout,
                         image)
        
class BrightnessTransform(object):
    def __init__(self, strength):
        """
        Change Brightness of image; pixel values 0 ~ 255 is assumed.
        
        Inputs:
            strength (float): determine strength of transform. 0 ~ 1.
        """
        self.strength = strength
        
    def __call__(self, rng, image):
        min_bright = 1 - self.strength
        max_bright = 1 + self.strength
        return jax.lax.clamp(0., image * jax.random.uniform(rng, shape=(1,), minval=min_bright, maxval=max_bright), 255.)

class ColorJitterTransform(object):
    def __init__(self, strength, prob=0.5):
        """
        Color Jittering of image; Change Hue , Saturation, Lightness (= brightness & contrast)
        pixel values 0. ~ 1. is assumed
        
        Inputs:
            strength (float): determine strength of transform. 0 ~ 1.
            prob (float): probability of the color jittering
        """
        self.strength = strength
        self.prob = prob
    
    def __call__(self, rng, image):
        min_strength = 1 - self.strength
        max_strength = 1 + self.strength
        
        rng, *keys = jax.random.split(rng, 6)
        
        # Brightness
        img_jt = image * jax.random.uniform(keys[0], shape=(1,), minval=min_strength, maxval=max_strength)  # Brightness
        img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
        
        # Contrast
        img_jt = dm_pix.random_contrast(keys[1], img_jt, lower=min_strength, upper=max_strength)
        img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
        
        # Saturation
        img_jt = dm_pix.random_saturation(keys[2], img_jt, lower=min_strength, upper=max_strength)
        img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
        
        # Hue
        img_jt = dm_pix.random_hue(keys[3], img_jt, max_delta=0.1)
        img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
        
        # Prob of color jitter
        return jnp.where(jax.random.bernoulli(keys[4], self.prob),
                         img_jt,
                         image)
