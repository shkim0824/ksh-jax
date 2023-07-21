import jax
import flax.linen as nn

__all__ = [
  "Classifier",
  "Classifier_transformer"
]

class Classifier(nn.Module):
  """
  Usual Classifier 
  """
  num_classes: int
  backbone: nn.Module
  
  @nn.compact
  def __call__(self, x):
    x = self.backbone(x)
    x = nn.Dense(
      self.num_classes, name='head')(x)
    return x

class Classifier_transformer(nn.Module):
  """
  Classifier that compatible to HuggingFace transformer backbone
  """
  num_classes: int
  backbone: nn.Module
  
  @nn.compact
  def __call__(self, x):
    x = self.backbone(x).pooler_output # we use pooler_output in transformer
    x = nn.Dense(
      self.num_classes, name='head', kernel_init=nn.zeros)(x) # To fine tuning, kernel_init=nn.zeros perform well
    return x
