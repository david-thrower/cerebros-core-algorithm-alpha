"""
Voxel-quantized output layer for Keras.

This layer simulates an output Dense layer whose kernel and bias are composed of
"voxels" (n_bits per weight/bias), decoded to a scalar via bit powers.

Key features:
- n_bits configurable (e.g., 1..8)
- Unsigned or signed range (centered) decoding
- Straight-Through Estimator (STE) for bit rounding during backprop
- Optional learnable per-unit scales to ease optimization

NOTE: This is a research-oriented layer; for production-grade quantization,
prefer Keras/TF quantization-aware training toolchains.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:  # Prefer standalone Keras if available
    import keras
except Exception:  # pragma: no cover - fallback for TF-bundled Keras
    from tensorflow import keras  # type: ignore

import tensorflow as tf


class VoxelQuantizedOutput(keras.layers.Layer):
    """
    Simulated quantized output layer with per-weight bit voxels.

    Args:
        units: Output dimensionality.
        n_bits: Number of bits per weight/bias voxel (>=1).
        activation: Activation applied to the layer output (string/callable/None).
                     Defaults to "sigmoid" to model independent outputs.
        use_bias: Whether to include a bias term.
        signed: If True, decode to a signed range roughly in
                [-(2^(n_bits-1)), 2^(n_bits-1)-1]. If False, range is [0, 2^n_bits-1].
        train_quant: If True, apply STE rounding of bits during training.
        learn_scales: Learn per-unit scales for kernel and bias to stabilize training.
        name: Optional layer name.
        dtype: Optional dtype policy.
        **kwargs: Standard Layer kwargs.
    """

    def __init__(
        self,
        units: int,
        n_bits: int = 4,
        activation: Optional[str | keras.layers.Activation | Any] = "sigmoid",
        use_bias: bool = True,
        signed: bool = False,
        train_quant: bool = True,
        learn_scales: bool = True,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, dtype=dtype, **kwargs)
        if n_bits < 1:
            raise ValueError("n_bits must be >= 1")
        if units < 1:
            raise ValueError("units must be >= 1")

        self.units = int(units)
        self.n_bits = int(n_bits)
        self.use_bias = bool(use_bias)
        self.signed = bool(signed)
        self.train_quant = bool(train_quant)
        self.learn_scales = bool(learn_scales)
        self.activation = keras.activations.get(activation)

        # Will be set in build()
        self.in_features: Optional[int] = None

    def build(self, input_shape: tf.TensorShape) -> None:
        input_dim = int(input_shape[-1])
        self.in_features = input_dim

        # Raw voxel parameters before binarization/quantization
        # Shape: [units, in_features, n_bits]
        self.weight_voxels = self.add_weight(
            name="weight_voxels",
            shape=(self.units, input_dim, self.n_bits),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            dtype=self.dtype,
        )

        if self.use_bias:
            # Shape: [units, n_bits]
            self.bias_voxels = self.add_weight(
                name="bias_voxels",
                shape=(self.units, self.n_bits),
                initializer=keras.initializers.Zeros(),
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias_voxels = None

        # Optional per-unit scales to help optimization dynamics
        if self.learn_scales:
            # weight_scale: [units, 1] so it broadcasts across in_features
            self.weight_scale = self.add_weight(
                name="weight_scale",
                shape=(self.units, 1),
                initializer=keras.initializers.Ones(),
                trainable=True,
                dtype=self.dtype,
            )
            if self.use_bias:
                self.bias_scale = self.add_weight(
                    name="bias_scale",
                    shape=(self.units,),
                    initializer=keras.initializers.Ones(),
                    trainable=True,
                    dtype=self.dtype,
                )
            else:
                self.bias_scale = None
        else:
            self.weight_scale = None
            self.bias_scale = None

        # Constant powers for decoding bits to integer values
        # bit_powers[k] = 2^k for k in [0..n_bits-1]
        self.bit_powers = tf.constant(
            [2 ** i for i in range(self.n_bits)], dtype=self.compute_dtype
        )

        # Offset for signed decoding (centered representation)
        if self.signed:
            self.signed_offset = tf.constant(2 ** (self.n_bits - 1), dtype=self.compute_dtype)
        else:
            self.signed_offset = None

        super().build(input_shape)

    @staticmethod
    def _hard_round_ste(x: tf.Tensor) -> tf.Tensor:
        """Round with straight-through estimator for gradients."""
        y = tf.round(x)
        return y + tf.stop_gradient(x - y)

    def _squash_to_bits(self, voxels: tf.Tensor, training: Optional[bool]) -> tf.Tensor:
        """
        Map raw voxel parameters to bit values in [0, 1] with optional STE rounding.
        """
        # Use a squashing function to map R -> (0,1)
        p = tf.sigmoid(voxels)
        if self.train_quant and (training or training is None):
            # STE rounding to {0,1}
            p = self._hard_round_ste(p)
        # Ensure numeric stability
        return tf.clip_by_value(p, 0.0, 1.0)

    def _decode_values(self, bit_tensor: tf.Tensor, *, is_bias: bool = False) -> tf.Tensor:
        """
        Decode a bit tensor to scalar values using bit powers and optional scales.

        For weights:  bit_tensor shape [units, in_features, n_bits] -> values [units, in_features]
        For bias:     bit_tensor shape [units, n_bits] -> values [units]
        """
        # Weighted sum over bits dimension
        values = tf.tensordot(bit_tensor, self.bit_powers, axes=[-1, 0])
        values = tf.cast(values, self.compute_dtype)

        # Signed centering if requested
        if self.signed:
            values = values - tf.cast(self.signed_offset, self.compute_dtype)

        # Apply optional scales
        if self.learn_scales:
            if is_bias:
                if self.use_bias and self.bias_scale is not None:
                    values = values * tf.cast(self.bias_scale, self.compute_dtype)
            else:
                if self.weight_scale is not None:
                    # weight_scale shape [units, 1] broadcasts over in_features
                    values = values * tf.cast(self.weight_scale, self.compute_dtype)

        return values

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # Quantize/binarize voxel params to bits
        weight_bits = self._squash_to_bits(self.weight_voxels, training=training)
        weight_values = self._decode_values(weight_bits, is_bias=False)

        if self.use_bias and self.bias_voxels is not None:
            bias_bits = self._squash_to_bits(self.bias_voxels, training=training)
            bias_values = self._decode_values(bias_bits, is_bias=True)
        else:
            bias_values = None

        # Linear transform: [B, D] x [units, D]^T -> [B, units]
        outputs = tf.linalg.matmul(inputs, tf.transpose(weight_values))
        if bias_values is not None:
            outputs = tf.nn.bias_add(outputs, bias_values)

        # Activation
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape((input_shape[0], self.units))

    def get_config(self) -> Dict[str, Any]:  # for serialization
        config: Dict[str, Any] = super().get_config()
        config.update(
            {
                "units": self.units,
                "n_bits": self.n_bits,
                "activation": keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "signed": self.signed,
                "train_quant": self.train_quant,
                "learn_scales": self.learn_scales,
            }
        )
        return config


__all__ = ["VoxelQuantizedOutput"]
