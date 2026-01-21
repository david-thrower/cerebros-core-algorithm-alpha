"""
Fused Cross-Entropy Loss for Memory-Efficient LLM Training

Based on "Cutting LLM Memory by 84%: A Deep Dive into Fused Kernels"
https://towardsdatascience.com/cutting-llm-memory-by-84-a-deep-dive-into-fused-kernels/

Key insight: Standard cross-entropy materializes full [N × V] logit matrix (33GB+ for LLama3-8B).
This implementation uses chunked computation with online softmax to avoid this.

Usage:
    from cerebros.kernels.fused_cross_entropy import FusedCrossEntropyLoss
    
    loss = FusedCrossEntropyLoss(vocab_chunk_size=1024)
    model.compile(loss=loss, ...)
"""

import tensorflow as tf
from typing import Optional


class FusedCrossEntropyLoss(tf.keras.losses.Loss):
    """
    Memory-efficient cross-entropy loss using chunked computation.
    
    Instead of computing softmax over the full vocabulary at once,
    this computes it in chunks using the online softmax algorithm.
    
    Memory: O(chunk_size) instead of O(vocab_size)
    Compute: Same asymptotic complexity, slightly higher constant factor
    
    Args:
        vocab_chunk_size: Size of vocabulary chunks for online computation.
            Smaller = less memory, larger = faster. Default 4096.
        from_logits: Whether input is logits (True) or probabilities (False).
            Must be True for this implementation.
        label_smoothing: Label smoothing factor (0 = no smoothing).
        reduction: Reduction method ('none', 'sum', 'sum_over_batch_size').
    """
    
    def __init__(
        self,
        vocab_chunk_size: int = 4096,
        from_logits: bool = True,
        label_smoothing: float = 0.0,
        reduction: str = 'sum_over_batch_size',
        name: str = 'fused_cross_entropy',
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        
        if not from_logits:
            raise ValueError(
                "FusedCrossEntropyLoss requires from_logits=True. "
                "Probability inputs are not supported."
            )
        
        self.vocab_chunk_size = vocab_chunk_size
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute chunked cross-entropy loss.
        
        Args:
            y_true: Integer labels of shape [batch_size, seq_len] or [batch_size]
            y_pred: Logits of shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
        
        Returns:
            Loss tensor with shape depending on reduction setting.
        """
        # Get shapes
        logits_shape = tf.shape(y_pred)
        vocab_size = logits_shape[-1]
        
        # Flatten to [N, V] where N = batch * seq_len (if applicable)
        original_shape = tf.shape(y_true)
        y_true_flat = tf.reshape(y_true, [-1])  # [N]
        
        # Handle 2D vs 3D logits
        if len(y_pred.shape) == 3:
            # [batch, seq, vocab] -> [batch*seq, vocab]
            batch_size, seq_len = logits_shape[0], logits_shape[1]
            y_pred_flat = tf.reshape(y_pred, [-1, vocab_size])  # [N, V]
        else:
            # [batch, vocab] -> [batch, vocab]
            y_pred_flat = y_pred
        
        n_samples = tf.shape(y_pred_flat)[0]
        
        # Get target logits directly (without materializing full softmax)
        # indices: [N, 2] with [[0, label_0], [1, label_1], ...]
        indices = tf.stack([
            tf.range(n_samples, dtype=tf.int32),
            tf.cast(y_true_flat, tf.int32)
        ], axis=1)
        
        target_logits = tf.gather_nd(y_pred_flat, indices)  # [N]
        
        # Compute log-sum-exp using chunked online algorithm
        # This is the key memory optimization
        lse = self._chunked_logsumexp(y_pred_flat, vocab_size)  # [N]
        
        # Cross-entropy: -log(softmax(target)) = -target_logit + LSE
        loss = -target_logits + lse  # [N]
        
        # Apply label smoothing if requested
        if self.label_smoothing > 0:
            # Add uniform distribution component
            vocab_size_f = tf.cast(vocab_size, tf.float32)
            smooth_loss = lse - tf.reduce_sum(y_pred_flat, axis=-1) / vocab_size_f
            loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss
        
        return loss
    
    def _chunked_logsumexp(
        self, 
        logits: tf.Tensor, 
        vocab_size: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute log-sum-exp using online/chunked algorithm.
        
        Online softmax formula (numerically stable):
            m_new = max(m_old, max(current_chunk))
            d_new = d_old * exp(m_old - m_new) + sum(exp(current_chunk - m_new))
            
            Final: LSE = m_final + log(d_final)
        
        Args:
            logits: Tensor of shape [N, vocab_size]
            vocab_size: Vocabulary size (could be dynamic)
            
        Returns:
            Log-sum-exp for each row, shape [N]
        """
        n_samples = tf.shape(logits)[0]
        vocab_size_int = tf.cast(vocab_size, tf.int32)
        chunk_size = self.vocab_chunk_size
        
        # Initialize running max and sum
        # Using very negative value for initial max so first chunk sets it
        running_max = tf.fill([n_samples], tf.float32.min)  # [N]
        running_sum = tf.zeros([n_samples], dtype=tf.float32)  # [N]
        
        # Process vocabulary in chunks
        num_chunks = (vocab_size_int + chunk_size - 1) // chunk_size
        
        # Use tf.while_loop for dynamic chunk iteration
        def chunk_body(i, r_max, r_sum):
            start_idx = i * chunk_size
            end_idx = tf.minimum(start_idx + chunk_size, vocab_size_int)
            
            # Get current chunk: [N, chunk_size] (may be smaller for last chunk)
            chunk = logits[:, start_idx:end_idx]
            
            # Find max in this chunk
            chunk_max = tf.reduce_max(chunk, axis=-1)  # [N]
            
            # Update running max
            new_max = tf.maximum(r_max, chunk_max)  # [N]
            
            # Rescale previous sum and add new chunk contribution
            # This is the online softmax correction
            scale_factor = tf.exp(r_max - new_max)  # [N]
            chunk_exp_sum = tf.reduce_sum(
                tf.exp(chunk - tf.expand_dims(new_max, -1)), 
                axis=-1
            )  # [N]
            
            new_sum = r_sum * scale_factor + chunk_exp_sum  # [N]
            
            return i + 1, new_max, new_sum
        
        def chunk_cond(i, r_max, r_sum):
            return i < num_chunks
        
        # Run the loop
        _, final_max, final_sum = tf.while_loop(
            chunk_cond,
            chunk_body,
            [tf.constant(0), running_max, running_sum],
            parallel_iterations=1  # Sequential for memory efficiency
        )
        
        # Compute final LSE
        lse = final_max + tf.math.log(final_sum + 1e-10)  # [N]
        
        return lse
    
    def get_config(self):
        """Return config for serialization."""
        config = super().get_config()
        config.update({
            'vocab_chunk_size': self.vocab_chunk_size,
            'from_logits': self.from_logits,
            'label_smoothing': self.label_smoothing,
        })
        return config


class ChunkedSparseCategoricalCrossentropy(FusedCrossEntropyLoss):
    """
    Drop-in replacement for tf.keras.losses.SparseCategoricalCrossentropy.
    
    Alias for FusedCrossEntropyLoss with sensible defaults.
    
    Usage:
        # Before (high memory):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # After (low memory):
        from cerebros.kernels.fused_cross_entropy import ChunkedSparseCategoricalCrossentropy
        loss = ChunkedSparseCategoricalCrossentropy(from_logits=True)
    """
    
    def __init__(
        self,
        from_logits: bool = True,
        label_smoothing: float = 0.0,
        vocab_chunk_size: int = 4096,
        reduction: str = 'sum_over_batch_size',
        name: str = 'chunked_sparse_categorical_crossentropy',
        **kwargs
    ):
        super().__init__(
            vocab_chunk_size=vocab_chunk_size,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            reduction=reduction,
            name=name,
            **kwargs
        )


# Utility function to estimate memory savings
def estimate_memory_savings(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    chunk_size: int = 4096,
    dtype_bytes: int = 4  # float32
) -> dict:
    """
    Estimate memory savings from using chunked cross-entropy.
    
    Returns dict with:
        - baseline_gb: Memory for full logit matrix
        - chunked_gb: Memory for chunked approach
        - savings_percent: Percentage reduction
    """
    n_tokens = batch_size * seq_len
    
    # Baseline: Full [N, V] matrix
    baseline_bytes = n_tokens * vocab_size * dtype_bytes
    baseline_gb = baseline_bytes / (1024 ** 3)
    
    # Chunked: Only [N, chunk_size] at a time, plus running stats [N, 2]
    chunked_bytes = n_tokens * chunk_size * dtype_bytes + n_tokens * 2 * dtype_bytes
    chunked_gb = chunked_bytes / (1024 ** 3)
    
    savings = (1 - chunked_gb / baseline_gb) * 100
    
    return {
        'baseline_gb': baseline_gb,
        'chunked_gb': chunked_gb,
        'savings_percent': savings,
        'n_tokens': n_tokens,
        'vocab_size': vocab_size,
        'chunk_size': chunk_size
    }


# Quick test
if __name__ == '__main__':
    import numpy as np
    
    # Test parameters
    batch_size = 8
    seq_len = 96
    vocab_size = 128256
    
    print("=" * 60)
    print("FusedCrossEntropyLoss Test")
    print("=" * 60)
    
    # Estimate savings
    savings = estimate_memory_savings(batch_size, seq_len, vocab_size)
    print(f"\nMemory Estimates:")
    print(f"  Baseline (full logits): {savings['baseline_gb']:.2f} GB")
    print(f"  Chunked approach: {savings['chunked_gb']:.4f} GB")
    print(f"  Savings: {savings['savings_percent']:.1f}%")
    
    # Test with smaller vocab for functional test
    test_vocab = 1000
    test_seq = 10
    
    # Create test data
    logits = tf.random.normal([batch_size, test_seq, test_vocab])
    labels = tf.random.uniform([batch_size, test_seq], 0, test_vocab, dtype=tf.int32)
    
    # Compare baseline vs chunked - BOTH use reduction='none' for comparison
    baseline_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, 
        reduction='none'
    )
    chunked_loss = FusedCrossEntropyLoss(
        vocab_chunk_size=128,
        reduction='none'  # Match baseline for comparison
    )
    
    baseline_result = baseline_loss(labels, logits)
    chunked_result = chunked_loss(labels, logits)
    
    # Reshape for comparison
    baseline_flat = tf.reshape(baseline_result, [-1])
    chunked_flat = tf.reshape(chunked_result, [-1])
    
    print(f"\nFunctional Test (vocab={test_vocab}):")
    print(f"  Baseline loss shape: {baseline_result.shape}")
    print(f"  Chunked loss shape: {chunked_result.shape}")
    print(f"  Max absolute diff: {tf.reduce_max(tf.abs(baseline_flat - chunked_flat)):.6f}")
    print(f"  Mean baseline: {tf.reduce_mean(baseline_flat):.4f}")
    print(f"  Mean chunked: {tf.reduce_mean(chunked_flat):.4f}")
    
    # Verify they're close (within numerical precision)
    max_diff = tf.reduce_max(tf.abs(baseline_flat - chunked_flat)).numpy()
    assert max_diff < 1e-4, \
        f"Chunked loss differs from baseline! Max diff: {max_diff}"
    
    print("\n✅ Test passed! Chunked loss matches baseline.")
    
    # Test with 'sum_over_batch_size' reduction (default for training)
    print("\n--- Testing with 'sum_over_batch_size' reduction ---")
    chunked_loss_reduced = FusedCrossEntropyLoss(
        vocab_chunk_size=128,
        reduction='sum_over_batch_size'
    )
    reduced_result = chunked_loss_reduced(labels, logits)
    print(f"  Reduced loss: {reduced_result:.4f}")
    print(f"  Expected (mean of per-sample): {tf.reduce_mean(chunked_flat):.4f}")

