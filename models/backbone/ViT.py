import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random, lax
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import transformers
from transformers.modeling_flax_outputs import *

ACT2FN = {"gelu": nn.activation.gelu}


class ViTConfig():
    """
    configuration class to store the configuration of 'ViTModel'
    """
    def __init__(
        self,
        hidden_size = 768,
        num_hidden_layers = 12, # number of ViTBlock; MHSA-MLP
        num_attention_heads = 12, # number of head of multi-head attention
        intermediate_size = 3072, # number of hidden units in MLP
        hidden_act = "gelu", # activation function
        hidden_dropout_prob = 0.0, # dropout prob of MLP
        attention_probs_dropout_prob = 0.0, # dropout prob of MHSA
        initializer_range = 0.02, # each dense layer is initialized by `initializer_range ** 2 * normal`
        layer_norm_eps = 1e-12, # hyper parameter for layer-normalization
        image_size = 224,
        patch_size = 16,
        num_channels = 3,
        qkv_bias = True,
        encoder_stride = 16
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride

class ViTPatchEmbeddings(nn.Module):
    """
    Patch Embedding of ViT
    
    Image를 patch size별로 나누어 kernel을 곱한다.
    이 과정은 stride가 patch size인 conv 연산으로 쉽게 구현 가능하다
    """
    config: ViTConfig
    
    def setup(self):
        image_size = self.config.image_size # default=224
        patch_size = self.config.patch_size # default=16
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.num_patches = num_patches
        self.num_channels = self.config.num_channels # default=3
        self.projection = nn.Conv(
            self.config.hidden_size,
            kernel_size = (patch_size, patch_size),
            strides = (patch_size, patch_size),
            padding="VALID",
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )
    
    def __call__(self, pixel_values):
        embeddings = self.projection(pixel_values)
        
        batch_size, _, _, channels = embeddings.shape
        return jnp.reshape(embeddings, (batch_size, -1, channels))

class ViTEmbeddings(nn.Module):
    """
    Embedding of ViT
    
    Create cls_token, do patch embeddings and add position embeddings
    """
    config: ViTConfig
    
    def setup(self):
        self.cls_token = self.param(
            "cls_token",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, 1, self.config.hidden_size),
        )
        self.patch_embeddings = ViTPatchEmbeddings(self.config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.param(
            "position_embeddings",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, num_patches + 1, self.config.hidden_size),
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
    
    def __call__(self, pixel_values, deterministic=True):
        batch_size = pixel_values.shape[0]
        
        embeddings = self.patch_embeddings(pixel_values) # patch embeddings
        
        # create cls_token and concatenate
        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)
        
        # add positional embeddings
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, deterministic=deterministic)
        
        return embeddings
    
class ViTModule(nn.Module):
    """
    Complete ViT Module
    
    pixel_value -> embeddings -> encoder -> layer norm -> pooling -> output
    """
    config: ViTConfig
        
    def setup(self):
        self.embeddings = ViTEmbeddings(self.config)
        self.encoder = ViTEncoder(self.config)
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps) # layer normalization for last layer
        self.pooler = ViTPooler(self.config)
    
    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False, # Boolean parameter to store attention of each ViT Block
        output_hidden_states: bool = False, # Boolean parameter to store hidden states of each ViT Block
        return_dict: bool = True,
    ):
        hidden_states = self.embeddings(pixel_values, deterministic=deterministic) # embedding image
        
        outputs = self.encoder(
            hidden_states,
            deterministic = deterministic,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict
        ) # outputs = `FlaxBaseModelOutput` class. outputs[0] = last hidden states
        
        hidden_states = outputs[0]
        hidden_states = self.layernorm(hidden_states)
        
        pooled = self.pooler(hidden_states)
        
        return FlaxBaseModelOutputWithPooling(
            last_hidden_state = hidden_states,
            pooler_output = pooled,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions
        )

class ViTEncoder(nn.Module):
    """
    ViT Encoder; just layer collection of ViT Block
    
    Exist for consistency of other Hugging Face modules
    """
    config: ViTConfig
    
    def setup(self):
        self.layer = ViTLayerCollection(self.config)
    
    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            deterministic = deterministic,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict
        )

class ViTLayerCollection(nn.Module):
    """
    ViT Layer Collection; collection of ViT Block
    """
    config: ViTConfig
    
    def setup(self):
        self.layers = [ViTLayer(self.config, name=str(i)) for i in range(self.config.num_hidden_layers)]
    
    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None # store attention of each ViT Block
        all_hidden_states = () if output_hidden_states else None # store hidden states of each ViT Block
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states, )
            
            layer_outputs = layer(hidden_states, deterministic=deterministic, output_attentions=output_attentions)
            
            hidden_states = layer_outputs[0] # layer output is (hidden_states, attention)
            
            if output_attentions:
                all_attentions += (layer_outputs[1], )
        
        if output_hidden_states:
            all_hidden_states += (hidden_states, )
        
        outputs = (hidden_states, )
        if not return_dict:
            return tuple(v for v in outputs if v is not None)
        
        return FlaxBaseModelOutput(
            last_hidden_state = hidden_states, hidden_states = all_hidden_states, attentions = all_attentions
        )

class ViTLayer(nn.Module):
    """
    ViT Block
    input -> layer norm -> multi-head attention -> layer norm -> MLP
    """
    config: ViTConfig
    
    def setup(self):
        self.attention = ViTAttention(self.config) # Multi-head attention
        self.intermediate = ViTIntermediate(self.config) # First MLP layer
        self.output = ViTOutput(self.config) # Second MLP layer
        self.layernorm_before = nn.LayerNorm(epsilon=self.config.layer_norm_eps) # Layer norm before MHSA
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps) # layer norm before MLP
    
    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        # Multi-head self-attention
        attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            deterministic=deterministic,
            output_attentions=output_attentions
        )
        
        attention_output = attention_outputs[0]
        
        # first residual connection
        attention_output = attention_output + hidden_states
        
        # MLP
        layer_output = self.layernorm_after(attention_output)
        
        hidden_states = self.intermediate(layer_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic) # second layer itself contain second residual connection
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (attention_outputs[1], )
        
        return outputs

class ViTAttention(nn.Module):
    """
    Multi-head self-attention layer of ViT
    
    Consists of self-attention and concatenating multi-head to one
    """
    config: ViTConfig
    
    def setup(self):
        self.attention = ViTSelfAttention(self.config) # self-attention
        self.output = ViTSelfOutput(self.config) # concatenating multi-head
    
    def __call__(self, hidden_states, deterministic=True, output_attentions:bool = False):
        attn_outputs = self.attention(hidden_states, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic) # original hidden_states exists just for consistency with other Hugging Face modules
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (attn_outputs[1], )
        
        return outputs
    
class ViTSelfAttention(nn.Module):
    """
    Self-attention of ViT
    """
    config: ViTConfig
    
    def setup(self):
        self.query = nn.Dense(
            self.config.hidden_size, # default=768
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias = self.config.qkv_bias
        )
        
        self.key = nn.Dense(
            self.config.hidden_size, # default=768
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias = self.config.qkv_bias
        )
        
        self.value = nn.Dense(
            self.config.hidden_size, # default=768
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias = self.config.qkv_bias
        )
    
    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        head_dim = self.config.hidden_size // self.config.num_attention_heads # dim for each head
        
        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        ) # shape[:2] means batch_size + number of token. We divide hidden dim to head-num * head_dim
        
        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = nn.attention.dot_product_attention_weights(
            query_states,
            key_states,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
    
class ViTSelfOutput(nn.Module):
    """
    Concatenating multi-head attention, with linear transform
    """
    config: ViTConfig
    
    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size, # default=768
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
    
    def __call__(self, hidden_states, input_tensor, deterministic: bool = True): # input_tensor exists just for consistency with other Hugging Face modules
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states

class ViTIntermediate(nn.Module):
    """
    First layer of ViT MLP
    """
    config: ViTConfig
        
    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class ViTOutput(nn.Module):
    """
    Second layer of ViT MLP
    
    Itself contains residual connection
    """
    config: ViTConfig
        
    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = hidden_states + attention_output
        return hidden_states

class ViTPooler(nn.Module):
    """
    ViT Pooler
    
    Image를 16 x 16을 하나의 patch로 자르면 여러개의 token이 생긴다.
    이 때 각 token마다 size (768, )의 hidden state가 생성된다.
    가령 224 x 224 size image는 196개의 token을 얻게 된다.
    이 때 연구에 의하면, 196개 token 마다 주어진 hidden state를 어떻게 잘 통합하여 사용하는 것보다
    cls token을 맨 앞에 추가하여 197개의 token을 input으로 넣고
    cls token에 대응되는 hidden state만 dense-tanh 변환을 거친 후 사용하여도 유사하거나 더 나은 성능을 보인다. 
    Pooler는 이 역할을 수행한다.
    """
    config: ViTConfig

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            )
        )

    def __call__(self, hidden_states):
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nn.tanh(cls_hidden_state)
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nn.tanh(cls_hidden_state)
