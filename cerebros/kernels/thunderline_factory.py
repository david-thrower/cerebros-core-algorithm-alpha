"""
Thunderline Stack Factory

Helper functions to create Thunderline-enhanced attention stacks.
Provides drop-in replacements for Cerebros attention blocks with:
- AIMBlock for O(n) attention
- OptimizedVoxelBlock with pruning
- Structured attention bias

Usage:
    from cerebros.kernels import create_thunderline_enhanced_stack
    
    model = create_thunderline_enhanced_stack(
        d_model=64,
        n_heads=8,
        n_layers=4,
        use_aim=True,
        use_optimized_voxel=True
    )

Author: Thunderline Engineering
Version: 1.0.0
"""

import tensorflow as tf
from cerebros.kernels.aim_block import AIMBlock
from cerebros.kernels.optimized_voxel import OptimizedVoxelBlock


def create_thunderline_enhanced_stack(
    d_model: int,
    n_heads: int = 8,
    n_layers: int = 4,
    dropout_rate: float = 0.1,
    structure_type: str = "grid",
    use_aim: bool = True,
    use_optimized_voxel: bool = True,
    ca_steps: int = 3,
    max_seq_len: int = 128
) -> tf.keras.Model:
    """
    Create a Thunderline-enhanced attention stack.
    
    Drop-in replacement for Cerebros attention blocks with:
    - AIMBlock for O(n) attention
    - OptimizedVoxelBlock with pruning
    - Structured attention bias
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of layers in stack
        dropout_rate: Dropout rate
        structure_type: Type of structural bias ("grid", "table", "tree")
        use_aim: Use AIMBlock for O(n) attention
        use_optimized_voxel: Use OptimizedVoxelBlock with pruning
        ca_steps: Number of CA evolution steps
        max_seq_len: Maximum sequence length
    
    Returns:
        Keras Model ready for integration
    """
    
    # Input
    inputs = tf.keras.Input(shape=(None, d_model))
    x = inputs
    
    # Build layers
    for i in range(n_layers):
        # AIM Block (or standard if disabled)
        if use_aim:
            x = AIMBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout_rate=dropout_rate,
                name=f"aim_block_{i}"
            )(x)
        
        # Optimized VoxelBlock
        if use_optimized_voxel:
            x = OptimizedVoxelBlock(
                d_model=d_model,
                dropout_rate=dropout_rate,
                ca_steps=ca_steps,
                name=f"optimized_voxel_{i}"
            )(x)
    
    # Output
    outputs = x
    
    model = tf.keras.Model(inputs, outputs, name="thunderline_enhanced_stack")
    
    return model
