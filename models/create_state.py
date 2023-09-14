def create_vit(rng, lr_fn, lr_fn2, lr_fn3, f, params=None):
    # Load ViT
    vit = transformers.FlaxViTModel.from_pretrained('google/vit-base-patch16-224-in21k') # Import pre-trained ViT in Hugging Face
    module = vit.module # Extract the Flax Module
    vit_vars = {'params': vit.params} # Extract the ViT parameters

    # Create Classifier and load pre-trained params
    model = Classifier(num_classes=n_targets, backbone=module) # Create classifier with ViT backbone, n_targets=10 classifier for CIFAR-10
    if params is None:
        variables = model.init(rng, jnp.ones((1, 224, 224, 3))) # Initialize classifier dense layer
        params = variables['params'] # Parameters of ViT + Dense
        params['backbone'] = vit_vars['params'] # Replace ViT to pre-trained parameter
        params = freeze(params)
        
    # print(jax.tree_util.tree_map(jnp.shape, unfreeze(params))) # print parameter dict name

    # Partition params which should be trained or not
    # linear probing
    # partition_optimizers = {'trainable': optax.adamw(learning_rate=lr_fn, weight_decay=0.01),
    #                         'frozen': optax.set_to_zero()} # We only train 'trainable' tag params
    
    # full fine tuning
#     partition_optimizers = {'trainable': optax.adamw(learning_rate=lr_fn, weight_decay=0.01), 
#                            'frozen': optax.adamw(learning_rate=lr_fn2, weight_decay=0.01),
#                            'semi-frozen': optax.adamw(learning_rate=lr_fn3, weight_decay=0.01)}
    
    # LP-FT
    partition_optimizers = {'trainable': lr_fn,
                            'frozen': lr_fn2,
                            'semi-frozen': lr_fn3}
    
    param_partitions = freeze(flax.traverse_util.path_aware_map(f, params))
    tx = optax.multi_transform(partition_optimizers, param_partitions) # optax.multi_transform match tag and apply different optimizers
    
    # Create TrainState
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    return state

def create_vit_adaptFormer(rng, lr_fn, lr_fn2, lr_fn3, f, params=None, adapter_size=64):
    # Load pre-trained ViT
    vit = transformers.FlaxViTModel.from_pretrained('google/vit-base-patch16-224-in21k') # Import pre-trained ViT in Hugging Face
    module = vit.module # Extract the Flax Module
    vit_vars = {'params': vit.params} # Extract the ViT parameters
    
    # Make ViT AdaptFormer and Load pre-trained weight
    config = ViTConfig(adapter_size=adapter_size)
    vit_adaptFormer = ViTModule(config)
    
    # Create Classifier and load pre-trained params
    model = Classifier(num_classes=n_targets, backbone=vit_adaptFormer) # Create classifier with ViT AdaptFormer backbone
    if params is None:
        variables = model.init(rng, jnp.ones((1, 224, 224, 3))) # Initialize classifier dense layer
        params = variables['params'] # Parameters of ViT + Dense

        def copy_leaf_values(source, target):
            for key, value in source.items():
                if isinstance(value, dict) and key in target:
                    copy_leaf_values(value, target[key])
                else:
                    target[key] = value
        copy_leaf_values(vit_vars['params'], params['backbone'])
        
        params = freeze(params)
        
    # print(jax.tree_util.tree_map(jnp.shape, unfreeze(params))) # print parameter dict name

    # Partition params which should be trained or not
    # linear probing
    # partition_optimizers = {'trainable': optax.adamw(learning_rate=lr_fn, weight_decay=0.01),
    #                         'frozen': optax.set_to_zero()} # We only train 'trainable' tag params
    
    # full fine tuning
#     partition_optimizers = {'trainable': optax.adamw(learning_rate=lr_fn, weight_decay=0.01), 
#                            'frozen': optax.adamw(learning_rate=lr_fn2, weight_decay=0.01),
#                            'semi-frozen': optax.adamw(learning_rate=lr_fn3, weight_decay=0.01)}
    
    # LP-FT
    partition_optimizers = {'trainable': lr_fn,
                            'frozen': lr_fn2,
                            'semi-frozen': lr_fn3}
    
    param_partitions = freeze(flax.traverse_util.path_aware_map(f, params))
    tx = optax.multi_transform(partition_optimizers, param_partitions) # optax.multi_transform match tag and apply different optimizers
    
    # Create TrainState
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    return state

def load_vit(params):
    # Create Classifier and load pre-trained params
    vit = transformers.FlaxViTModel.from_pretrained('google/vit-base-patch16-224-in21k') # Import pre-trained ViT in Hugging Face
    module = vit.module # Extract the Flax Module
    model = Classifier(num_classes=n_targets, backbone=module) # Create classifier with ViT backbone, n_targets=10 classifier for CIFAR-10

    # Partition params which should be trained or not
    partition_optimizers = {'trainable': optax.adamw(learning_rate=scheduler_cos(), weight_decay=0.01), 'frozen': optax.set_to_zero()} # We only train 'trainable' tag params
    param_partitions = freeze(flax.traverse_util.path_aware_map(
        lambda path, v: 'frozen' if 'backbone' in path else 'trainable', params)) # Partition params that should be 'trainable' or not
    tx = optax.multi_transform(partition_optimizers, param_partitions) # optax.multip_transform match tag and apply different optimizers
    
    # Create TrainState
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    return state

def load_vit_adaptFormer(params, adapter_size=64):
    # Make ViT AdaptFormer and Load pre-trained weight
    config = ViTConfig(adapter_size=adapter_size)
    vit_adaptFormer = ViTModule(config)
    
    # Create Classifier and load pre-trained params
    model = Classifier(num_classes=n_targets, backbone=vit_adaptFormer) # Create classifier with ViT AdaptFormer backbone
    
    # Partition params which should be trained or not
    partition_optimizers = {'trainable': optax.adamw(learning_rate=scheduler_cos(), weight_decay=0.01), 'frozen': optax.set_to_zero()} # We only train 'trainable' tag params
    param_partitions = freeze(flax.traverse_util.path_aware_map(
        lambda path, v: 'frozen' if 'backbone' in path else 'trainable', params)) # Partition params that should be 'trainable' or not
    tx = optax.multi_transform(partition_optimizers, param_partitions) # optax.multip_transform match tag and apply different optimizers
    
    # Create TrainState
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    return state
