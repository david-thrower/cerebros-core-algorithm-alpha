# Thunderline-Cerebros Integration: XLA SplitV Dynamic Shape Issue

## TL;DR
**Phase I-b training fails with XLA SplitV compile-time constant error.** The workaround is `run_eagerly=True`, but this disables graph optimization. Root cause is dynamic shape operations in Cerebros NAS `FinalDenseUnit` concatenation layer.

---

## What Works
- ✅ **Thunderline OptimizedVoxelBlock** integrated as drop-in replacement for VoxelBlock
- ✅ **Keras serialization** fixed with `@register_keras_serializable` decorators
- ✅ **Phase I-a training** completes successfully (perplexity drops from 128k → 60k in 5 epochs)
- ✅ **Model save/load** works after serialization fix

## What Fails
- ❌ **Phase I-b training** crashes with XLA SplitV error when using graph compilation
- ❌ Error occurs in `FinalDenseUnit` concatenation layer (Cerebros NAS code, not Thunderline)

---

## The Error

```
tensorflow.python.framework.errors_impl.InvalidArgumentError: 
Input 1 to node `gradient_tape/.../FinalDenseUnit.../split` with op SplitV must be a compile-time constant.

XLA compilation requires that operator arguments that represent shapes or dimensions 
be evaluated to concrete values at compile time.
```

**Stack trace points to:**
```
cerebros/.../FinalDenseUnit_0000000000000002_tr_0_0_cat__1/split
```

This is the Cerebros NAS dynamic topology code, not Thunderline.

---

## Root Cause Analysis

The Cerebros NAS creates dynamic neural architectures with:
1. Variable-width concatenation layers (`FinalDenseUnit...cat__1`)
2. Skip connections with dynamic shapes
3. `tf.split()` operations where split sizes are computed at runtime

XLA/JIT requires static shapes at graph compile time. When split sizes depend on:
- Model topology (determined by NAS)
- Runtime tensor dimensions
- Dynamic skip connection routing

...XLA cannot compile the graph.

---

## Current Workaround

```python
# Before Phase I-b fit():
generator.model.compile(
    optimizer=generator.model.optimizer,
    loss=generator.model.loss,
    metrics=generator.model.metrics,
    run_eagerly=True,   # <-- Disables graph compilation
    jit_compile=False
)
```

**Downsides:**
- Slower execution (no graph optimization)
- Still uses XLA for some ops (TF 2.20 behavior)

---

## Files Changed

| File | Change |
|------|--------|
| `thunderline_integration.py` | Added `@register_keras_serializable` decorators |
| `train_a_generative_llm_docker.py` | Added `run_eagerly=True` recompile before Phase I-b |
| `train_a_generative_llm_docker.py` | Added XLA disable env vars before TF import |

---

## Recommended Fixes (for Cerebros team)

### Option A: Static Shape Assertions
In `FinalDenseUnit`, replace dynamic splits with static padded tensors:
```python
# Instead of:
tf.split(x, num_splits)  # where num_splits is dynamic

# Use:
x_padded = tf.pad(x, [[0,0], [0, max_size - current_size]])
# then slice statically
```

### Option B: Mark Layers as Non-JIT
```python
@tf.function(jit_compile=False)
def call(self, inputs):
    # ...existing code...
```

### Option C: Use tf.TensorArray
For dynamic collections instead of split/concat.

---

## Test Commands

```bash
# Full CPU test with XLA workarounds:
cd /home/mo/2026-01-16-final-cpu-hpo-run/cerebros-309-docker

TF_XLA_FLAGS="--tf_xla_auto_jit=0" \
XLA_FLAGS="--xla_gpu_cuda_data_dir=" \
CUDA_VISIBLE_DEVICES="" \
ARTIFACTS_FOLDER="$(pwd)/artifacts" \
python train_a_generative_llm_docker.py
```

---

## Environment

- TensorFlow: 2.20.0
- Python: 3.12
- Keras: 3.x (standalone)
- Platform: Fedora Linux, CPU (RTX 5080 not compatible with TF 2.20)

---

## Questions for El Tigere

1. Is the dynamic `split` in `FinalDenseUnit` intentional for architecture flexibility?
2. Can we pre-compute split sizes during model construction and store as constants?
3. Is there a reason Phase I-a works but Phase I-b doesn't? (Different model compilation path?)
