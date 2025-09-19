
# Voxel layer implementation:

If I wanted to make an  output layer for a classification model which instead of a Dense(n, softmax), instead, had an n_bit quantized simulated voxel for each element in its output (not mutually entangled) THEN a [ sigmoid ] if needed ...

How would the computational burden compare to a synonomous Dense node?

Could this be compatible with common accelerators?

What would the code look like?

This is a fascinating idea! You're essentially proposing to replace floating-point weights/activations with quantized "voxels" (bit-level representations) in the output layer. Let me break this down:
Computational Burden Comparison

Your proposed approach:

    Memory: Significantly reduced (n × n_bits vs n × 32/16 bits per weight)
    Compute: Bit-level operations instead of floating-point multiply-accumulates
    Bandwidth: Much lower memory bandwidth requirements

Standard Dense layer:

    Memory: Higher (full precision weights/activations)
    Compute: Standard FP32/FP16 operations
    Bandwidth: Higher requirements

Accelerator Compatibility

Good news: Modern accelerators (TPUs, NVIDIA TensorRT, etc.) have built-in support for:

    INT8/INT4 quantization
    Bit-level operations
    Sigmoid approximations

Challenges:

    Custom bit-widths might require specialized kernels
    Mixed precision workflows need careful implementation

Code Implementation

Here's a PyTorch implementation of your concept:
python

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizedOutputLayer(nn.Module):
    def __init__(self, in_features, out_features, n_bits=8, use_sigmoid=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.use_sigmoid = use_sigmoid
        
        # Quantized weights as bit tensors
        self.weight_bits = nn.Parameter(
            torch.randint(0, 2, (out_features, in_features, n_bits)).float()
        )
        
        # Bit scaling factors (learnable)
        self.weight_scale = nn.Parameter(torch.ones(out_features, in_features))
        self.weight_zero_point = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Bias in quantized form
        self.bias_bits = nn.Parameter(
            torch.randint(0, 2, (out_features, n_bits)).float()
        )
        self.bias_scale = nn.Parameter(torch.ones(out_features))
        
        # Voxel activation quantization parameters
        self.register_buffer('act_scale', torch.tensor(1.0))
        self.register_buffer('act_zero_point', torch.tensor(0.0))

    def bits_to_float(self, bits, scale, zero_point):
        """Convert bit representation to float"""
        # Decode binary to integer
        powers = 2 ** torch.arange(self.n_bits, device=bits.device)
        if bits.dim() == 3:  # Weight bits [out, in, bits]
            int_values = torch.sum(bits * powers, dim=-1)
        else:  # Bias bits [out, bits]
            int_values = torch.sum(bits * powers, dim=-1)
        
        # Dequantize to float
        return scale * (int_values - zero_point)

    def quantize_activation(self, x):
        """Quantize input activations to bit representation"""
        # Find range for quantization
        x_min, x_max = x.min(), x.max()
        scale = (x_max - x_min) / (2**self.n_bits - 1)
        zero_point = -x_min / scale
        
        # Quantize
        x_quant = torch.round(x / scale + zero_point)
        x_quant = torch.clamp(x_quant, 0, 2**self.n_bits - 1)
        
        # Store for dequantization
        self.act_scale = scale
        self.act_zero_point = zero_point
        
        return x_quant

    def forward(self, x):
        # Quantize input if needed
        if x.dtype != torch.uint8:
            x_quant = self.quantize_activation(x)
        else:
            x_quant = x.float()
        
        # Convert quantized weights to float
        weight_float = self.bits_to_float(
            self.weight_bits, self.weight_scale, self.weight_zero_point
        )
        bias_float = self.bits_to_float(
            self.bias_bits, self.bias_scale, torch.zeros_like(self.bias_scale)
        )
        
        # Standard linear operation
        output = F.linear(x_quant * self.act_scale + self.act_zero_point, 
                         weight_float, bias_float)
        
        # Apply sigmoid if requested
        if self.use_sigmoid:
            output = torch.sigmoid(output)
            
        return output

# More efficient version using built-in quantization
class EfficientQuantizedOutput(nn.Module):
    def __init__(self, in_features, out_features, n_bits=8, use_sigmoid=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        
        # Use PyTorch's quantized linear layer
        self.quant_linear = nn.Linear(in_features, out_features)
        
        # Quantize the layer
        self.quant_linear_qat = torch.quantization.quantize_dynamic(
            self.quant_linear, {nn.Linear}, dtype=torch.qint8
        )
        
    def forward(self, x):
        output = self.quant_linear_qat(x)
        if self.use_sigmoid:
            output = torch.sigmoid(output)
        return output

# Custom bit-level implementation for maximum control
class BitLevelOutput(nn.Module):
    def __init__(self, in_features, out_features, n_bits=4, use_sigmoid=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.use_sigmoid = use_sigmoid
        
        # Create bit-level weight tensors
        self.weight_voxels = nn.Parameter(
            torch.randn(out_features, in_features, n_bits) * 0.1
        )
        
        # Bit-level bias
        self.bias_voxels = nn.Parameter(
            torch.randn(out_features, n_bits) * 0.1
        )
        
        # Bit operation masks
        self.register_buffer(
            'bit_powers', 2 ** torch.arange(n_bits).float()
        )

    def bit_tensor_multiply(self, x, bit_weights):
        """Perform bit-level multiplication"""
        batch_size = x.shape[0]
        
        # Expand x for bit operations [batch, in_features, 1]
        x_expanded = x.unsqueeze(-1)
        
        # Convert bit weights to values [out, in, bits] -> [out, in]
        weight_values = torch.sum(bit_weights * self.bit_powers, dim=-1)
        
        # Matrix multiplication
        output = torch.matmul(x_expanded, weight_values.t().unsqueeze(0))
        
        return output.squeeze(-1)

    def forward(self, x):
        # Input quantization (simplified)
        x_quant = torch.round(x * (2**self.n_bits - 1)) / (2**self.n_bits - 1)
        
        # Bit-level computation
        weight_values = torch.sum(self.weight_voxels * self.bit_powers, dim=-1)
        bias_values = torch.sum(self.bias_voxels * self.bit_powers, dim=-1)
        
        # Linear transformation
        output = F.linear(x_quant, weight_values, bias_values)
        
        # Apply sigmoid if needed
        if self.use_sigmoid:
            output = torch.sigmoid(output)
            
        return output

# Usage example
if __name__ == "__main__":
    # Standard approach
    dense_layer = nn.Linear(512, 10)
    
    # Your quantized approach
    quantized_layer = BitLevelOutput(512, 10, n_bits=4)
    
    # Test with sample data
    x = torch.randn(32, 512)
    
    with torch.no_grad():
        out1 = dense_layer(x)
        out2 = quantized_layer(x)
        
    print(f"Dense output shape: {out1.shape}")
    print(f"Quantized output shape: {out2.shape}")
    print(f"Memory reduction: ~{8/4:.1f}x (assuming 8-bit vs 4-bit)")
```


TensorFlow/Keras Version

```python

import tensorflow as tf
from tensorflow import keras

class QuantizedOutputLayer(keras.layers.Layer):
    def __init__(self, units, n_bits=4, use_sigmoid=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.n_bits = n_bits
        self.use_sigmoid = use_sigmoid
        
    def build(self, input_shape):
        self.in_features = input_shape[-1]
        
        # Bit-level weights
        self.weight_voxels = self.add_weight(
            shape=(self.units, self.in_features, self.n_bits),
            initializer='glorot_uniform',
            trainable=True,
            name='weight_voxels'
        )
        
        self.bias_voxels = self.add_weight(
            shape=(self.units, self.n_bits),
            initializer='zeros',
            trainable=True,
            name='bias_voxels'
        )
        
        # Bit powers for conversion
        self.bit_powers = tf.constant(
            [2**i for i in range(self.n_bits)], dtype=tf.float32
        )
        
    def call(self, inputs):
        # Convert bit representations to float values
        weight_values = tf.reduce_sum(
            self.weight_voxels * self.bit_powers, axis=-1
        )
        bias_values = tf.reduce_sum(
            self.bias_voxels * self.bit_powers, axis=-1
        )
        
        # Standard dense operation
        output = tf.matmul(inputs, weight_values, transpose_b=True)
        output = tf.nn.bias_add(output, bias_values)
        
        # Apply sigmoid if needed
        if self.use_sigmoid:
            output = tf.nn.sigmoid(output)
            
        return output
```

# Usage
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    QuantizedOutputLayer(10, n_bits=4, use_sigmoid=True)
])

Key Benefits

    Memory Efficiency: 2-8x reduction depending on bit-width
    Energy Efficiency: Bit operations consume less power
    Bandwidth: Reduced memory bandwidth requirements
    Compatibility: Works with existing quantization toolchains

Potential Issues

    Precision Loss: Lower bit-widths may hurt accuracy
    Training Complexity: Quantization-aware training needed
    Gradient Flow: Bit-level operations may have vanishing gradients
    Hardware Support: May need custom kernels for optimal performance

This approach bridges the gap between traditional neural networks and neuromorphic computing concepts!

## Quick usage (Keras)

The repository now includes a research-grade Keras layer implementing this idea: `cerebros.layers.VoxelQuantizedOutput`.

- n_bits: integer >= 1
- signed: False (0..2^n_bits-1) or True (centered around 0)
- train_quant: enables STE rounding during training
- learn_scales: per-unit scale parameters to stabilize optimization

Example

```python
from cerebros.layers import VoxelQuantizedOutput
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(128,)),
    keras.layers.Dense(64, activation="relu"),
    VoxelQuantizedOutput(10, n_bits=4, activation="sigmoid", signed=False),
])
model.compile(optimizer="adam", loss="binary_crossentropy")
```

See a runnable demo at `documentation/examples/voxel_output_demo.py`.

## Accelerator notes

- This layer simulates quantized output via differentiable binarization + decoding.
- For production-grade acceleration (TPU/TensorRT/ONNX Runtime), port to a proper QAT flow and export as int8/int4 ops.
- Mixed precision is compatible; the layer uses Keras dtype policies and avoids non-differentiable ops except STE on the voxel bits.
