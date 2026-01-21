"""
Cerebros Kernels Package

Memory-efficient and fused kernel implementations for LLM training.

## Loss Functions
- FusedCrossEntropyLoss: Memory-efficient cross-entropy with chunking
- ChunkedSparseCategoricalCrossentropy: Drop-in replacement for Keras loss

## Thunderline Research Kernels
- AIMBlock: O(n) Hadamard-fused attention (P3)
- TensorTreeOptimizer: Dynamic pruning for 40-60% compute savings (P6)
- StructuredAttentionBias: Structure-aware attention biases (P4)
- OptimizedVoxelBlock: VoxelBlock with integrated pruning
- create_thunderline_enhanced_stack: Factory for Thunderline-enhanced stacks
"""

from cerebros.kernels.fused_cross_entropy import (
    FusedCrossEntropyLoss,
    ChunkedSparseCategoricalCrossentropy,
    estimate_memory_savings,
)

from cerebros.kernels.aim_block import AIMBlock
from cerebros.kernels.tensor_tree import TensorTreeOptimizer
from cerebros.kernels.structured_attention import StructuredAttentionBias
from cerebros.kernels.optimized_voxel import OptimizedVoxelBlock
from cerebros.kernels.thunderline_factory import create_thunderline_enhanced_stack

__all__ = [
    # Loss functions
    'FusedCrossEntropyLoss',
    'ChunkedSparseCategoricalCrossentropy',
    'estimate_memory_savings',
    # Thunderline kernels
    'AIMBlock',
    'TensorTreeOptimizer',
    'StructuredAttentionBias',
    'OptimizedVoxelBlock',
    'create_thunderline_enhanced_stack',
]
