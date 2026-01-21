"""
Structured Attention Bias for Structure-Aware Learning

Based on Thunderline P4 research: Structure-aware bias injection.
Adds structure-aware bias to attention mechanisms for:
- Table structure (row/column membership)
- Grid structure (CA lattice distance)
- Tree structure (hierarchical level)

Key benefit: Model doesn't waste capacity learning position relationships

Usage:
    from cerebros.kernels import StructuredAttentionBias
    
    struct_bias = StructuredAttentionBias(
        d_model=64,
        structure_type="grid",
        max_seq_len=128
    )
    biased_scores = struct_bias(attention_scores)

Author: Thunderline Engineering
Version: 1.0.0
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='cerebros.kernels', name='StructuredAttentionBias')
class StructuredAttentionBias(tf.keras.layers.Layer):
    """
    Structured Attention Bias Layer.
    
    Adds structure-aware bias to attention mechanisms based on:
    - Table structure (row/column membership)
    - Grid structure (CA lattice distance)
    - Tree structure (hierarchical level)
    
    Key benefit: Model doesn't waste capacity learning position relationships
    
    Usage:
        struct_bias = StructuredAttentionBias(
            d_model=64,
            structure_type="grid",
            max_seq_len=128
        )
        biased_attention = struct_bias(attention_scores, positions)
    """
    
    def __init__(
        self,
        d_model: int,
        structure_type: str = "grid",  # "table", "grid", or "tree"
        max_seq_len: int = 1024,
        decay_rate: float = 0.1,
        n_heads: int = 8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.structure_type = structure_type
        self.max_seq_len = max_seq_len
        self.decay_rate = decay_rate
        self.n_heads = n_heads
    
    def build(self, input_shape):
        # Learnable bias parameters
        self.structure_scale = self.add_weight(
            name="structure_scale",
            shape=(self.n_heads, 1),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )
        
        self.distance_decay = self.add_weight(
            name="distance_decay",
            shape=(self.n_heads, 1),
            initializer=tf.keras.initializers.Constant(self.decay_rate),
            trainable=True
        )
        
        # Pre-compute position indices
        positions = tf.range(self.max_seq_len, dtype=tf.float32)
        self.position_matrix = tf.abs(
            tf.expand_dims(positions, 0) - tf.expand_dims(positions, 1)
        )
        
        super().build(input_shape)
    
    def compute_grid_bias(self, seq_len):
        """
        Compute grid-based attention bias.
        
        For CA lattices: closer grid positions get higher attention bias.
        """
        # Get relevant portion of position matrix
        pos_matrix = self.position_matrix[:seq_len, :seq_len]
        
        # Distance decay: exp(-decay * distance)
        # decay shape: (n_heads, 1) -> need (n_heads, 1, 1) for broadcast
        decay = tf.abs(self.distance_decay)  # (n_heads, 1)
        decay = tf.reshape(decay, (self.n_heads, 1, 1))  # (n_heads, 1, 1)
        
        # pos_matrix: (seq, seq) -> (1, seq, seq) for broadcast
        pos_expanded = tf.expand_dims(pos_matrix, 0)  # (1, seq, seq)
        
        # Compute bias: (n_heads, seq, seq)
        bias = tf.exp(-decay * pos_expanded)
        
        # Scale by learned parameter: scale is (n_heads, 1) -> (n_heads, 1, 1)
        scale = tf.reshape(self.structure_scale, (self.n_heads, 1, 1))
        bias = bias * scale
        
        return bias
    
    def compute_table_bias(self, seq_len, row_size: int = 8):
        """
        Compute table-based attention bias.
        
        Positions in same row/column get bias boost.
        """
        positions = tf.range(seq_len, dtype=tf.float32)
        
        # Compute row and column indices
        rows = tf.cast(positions // row_size, tf.float32)
        cols = tf.cast(positions % row_size, tf.float32)
        
        # Same row indicator
        same_row = tf.cast(
            tf.equal(
                tf.expand_dims(rows, 0),
                tf.expand_dims(rows, 1)
            ),
            tf.float32
        )
        
        # Same column indicator
        same_col = tf.cast(
            tf.equal(
                tf.expand_dims(cols, 0),
                tf.expand_dims(cols, 1)
            ),
            tf.float32
        )
        
        # Combined bias: (seq, seq)
        bias = same_row + same_col
        
        # Add head dimension: (1, seq, seq) -> broadcast to (n_heads, seq, seq)
        # Scale is (n_heads, 1) -> reshape to (n_heads, 1, 1)
        scale = tf.reshape(self.structure_scale, (self.n_heads, 1, 1))
        bias = tf.expand_dims(bias, 0) * scale
        
        return bias
    
    def compute_tree_bias(self, seq_len, branching: int = 2):
        """
        Compute tree-based attention bias.
        
        Positions at same tree level or with common ancestors get bias.
        """
        positions = tf.range(seq_len, dtype=tf.float32)
        
        # Compute tree level (log2 approximation)
        levels = tf.math.floor(tf.math.log(positions + 1.0) / tf.math.log(2.0))
        
        # Same level indicator: (seq, seq)
        same_level = tf.cast(
            tf.equal(
                tf.expand_dims(levels, 0),
                tf.expand_dims(levels, 1)
            ),
            tf.float32
        )
        
        # Level distance penalty
        level_dist = tf.abs(
            tf.expand_dims(levels, 0) - tf.expand_dims(levels, 1)
        )
        
        # Reshape decay and scale for broadcast: (n_heads, 1) -> (n_heads, 1, 1)
        decay = tf.reshape(self.distance_decay, (self.n_heads, 1, 1))
        scale = tf.reshape(self.structure_scale, (self.n_heads, 1, 1))
        
        # level_dist: (seq, seq) -> (1, seq, seq)
        level_dist_exp = tf.expand_dims(level_dist, 0)
        level_penalty = tf.exp(-decay * level_dist_exp)
        
        # same_level: (seq, seq) -> (1, seq, seq) -> broadcast with (n_heads, 1, 1)
        same_level_exp = tf.expand_dims(same_level, 0)
        
        # Combined bias: (n_heads, seq, seq)
        bias = same_level_exp + level_penalty * scale
        
        return bias
    
    def call(self, attention_scores, training=None):
        """
        Add structural bias to attention scores.
        
        Args:
            attention_scores: (batch, heads, seq, seq) attention logits
            training: Whether in training mode
            
        Returns:
            Biased attention scores
        """
        seq_len = tf.shape(attention_scores)[2]
        
        # Compute appropriate bias
        if self.structure_type == "grid":
            bias = self.compute_grid_bias(seq_len)
        elif self.structure_type == "table":
            bias = self.compute_table_bias(seq_len)
        elif self.structure_type == "tree":
            bias = self.compute_tree_bias(seq_len)
        else:
            # No bias
            return attention_scores
        
        # Ensure bias has correct shape for broadcasting
        # bias: (heads, seq, seq) -> (1, heads, seq, seq)
        bias = tf.expand_dims(bias, 0)
        
        # Slice to match actual sequence length
        bias = bias[:, :, :seq_len, :seq_len]
        
        # Add bias to attention scores
        biased_scores = attention_scores + bias
        
        return biased_scores
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "structure_type": self.structure_type,
            "max_seq_len": self.max_seq_len,
            "decay_rate": self.decay_rate,
            "n_heads": self.n_heads
        })
        return config
