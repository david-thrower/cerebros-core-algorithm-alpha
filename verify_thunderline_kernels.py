#!/usr/bin/env python3
"""
Thunderline Kernels Verification Script

Tests all newly integrated Thunderline kernel modules for:
1. Clean imports
2. Instantiation and forward pass
3. Serialization round-trip

Author: Thunderline Engineering
"""

import sys
sys.path.insert(0, '/home/mo/2026-01-16-final-cpu-hpo-run/cerebros-core-algorithm-alpha')

import tensorflow as tf
import tempfile
import os

print("=" * 70)
print("THUNDERLINE KERNELS VERIFICATION")
print("=" * 70)

# Test 1: Import Verification
print("\n[TEST 1] Import Verification")
print("-" * 70)
try:
    from cerebros.kernels import (
        AIMBlock,
        TensorTreeOptimizer,
        StructuredAttentionBias,
        OptimizedVoxelBlock,
        create_thunderline_enhanced_stack
    )
    print("✅ All Thunderline kernels imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Module Instantiation
print("\n[TEST 2] Module Instantiation & Forward Pass")
print("-" * 70)

try:
    # Test AIMBlock
    aim = AIMBlock(d_model=64, n_heads=8)
    x = tf.random.normal((2, 32, 64))
    out = aim(x)
    assert out.shape == x.shape, "AIMBlock output shape mismatch"
    print("✅ AIMBlock instantiation and forward pass OK")
    
    # Test TensorTreeOptimizer
    tto = TensorTreeOptimizer(threshold=0.1)
    out = tto(x)
    stats = tto.get_pruning_stats()
    assert "pruned_ratio" in stats, "TensorTree stats missing"
    print(f"✅ TensorTreeOptimizer OK (pruned {stats['pruned_ratio']:.2%})")
    
    # Test OptimizedVoxelBlock
    ovb = OptimizedVoxelBlock(d_model=64, pruning_threshold=0.1)
    out = ovb(x)
    assert out.shape == x.shape, "OptimizedVoxelBlock output shape mismatch"
    print("✅ OptimizedVoxelBlock OK")
    
    # Test StructuredAttentionBias
    sab = StructuredAttentionBias(d_model=64, structure_type="grid", n_heads=8)
    # Create dummy attention scores (batch, heads, seq, seq)
    attn_scores = tf.random.normal((2, 8, 32, 32))
    biased = sab(attn_scores)
    assert biased.shape == attn_scores.shape, "StructuredAttentionBias shape mismatch"
    print("✅ StructuredAttentionBias OK")
    
    # Test factory
    model = create_thunderline_enhanced_stack(
        d_model=64,
        n_heads=8,
        n_layers=2,
        use_aim=True,
        use_optimized_voxel=True
    )
    test_out = model(x)
    print(f"✅ Factory OK: created stack with {len(model.layers)} layers")
    
except Exception as e:
    print(f"❌ Instantiation test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Serialization Round-Trip
print("\n[TEST 3] Serialization Round-Trip")
print("-" * 70)

try:
    # Create model with OptimizedVoxelBlock
    inputs = tf.keras.Input(shape=(32, 64))
    x = OptimizedVoxelBlock(d_model=64, pruning_threshold=0.1)(inputs)
    model = tf.keras.Model(inputs, x)
    
    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.keras")
        model.save(path)
        print(f"✅ Model saved to {path}")
        
        # Load
        loaded = tf.keras.models.load_model(path)
        print("✅ Model loaded successfully")
        
        # Test forward pass
        test_input = tf.random.normal((2, 32, 64))
        original_out = model(test_input)
        loaded_out = loaded(test_input)
        
        diff = tf.reduce_max(tf.abs(original_out - loaded_out))
        assert diff < 1e-5, f"Loaded model outputs differ: {diff}"
        print(f"✅ Serialization round-trip OK (max diff: {diff:.2e})")
    
except Exception as e:
    print(f"❌ Serialization test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("🎯 ALL KERNEL MODULES VERIFIED SUCCESSFULLY!")
print("=" * 70)
print("\nNext steps:")
print("1. Update imports in train_a_generative_llm_docker.py")
print("2. Replace VoxelBlock with OptimizedVoxelBlock")
print("3. Run training with Thunderline kernels")
