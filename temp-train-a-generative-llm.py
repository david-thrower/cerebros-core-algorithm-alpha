import tensorflow as tf
from transformers import AutoTokenizer


from cerebrosllmutils.llm_utils import (
    InterleavedRoPE,
    GatedMergeLayer,
    ChunkedAttentionBlock,
    MambaBlock,
    VoxelBlock,
    LinformerBlock,
    AdapterBlock
)



# --- Core Model Constants ---
MAX_SEQ_LENGTH = 40  # Example value, adjust as needed

# Tokenization

tokenizer_checkpoint = "HuggingFaceTB/SmolLM3-3B"  # "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

# Step 1: Add special tokens
special_tokens = {
    "additional_special_tokens": ["<prompt>", "</prompt>", "<response>", "</response>"]
}
tokenizer.add_special_tokens(special_tokens)

VOCABULARY_SIZE = len(tokenizer)

# For interleaved Rotary Positional Embedding (iRoPE), the
# embedding output dim must be an even number
# Maximize EMBEDDING_N based on available RAM and CPU / GPU
EMBEDDING_N = 6 # trial.suggest_int('embedding_n',6, 9) # 12
# Don't change directly. Use EMBEDDING_N to control
EMBEDDING_DIM = int(EMBEDDING_N * 2)


# --- Initial Stream Merging & Dropout ---
POSITIONAL_EMBEDDING_DROPOUT = 0.1

# --- SingleHeadChunkedAttention Block Constants ---
K_PROJ_CHUNKED = 5
DFF_CHUNKED = EMBEDDING_DIM # Can be tuned independently, but likely to coincide.
DROPOUT_RATE_CHUNKED = 0.1

# --- MAMBA Block Constants ---
MAMBA_D_STATE = 12
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2
MAMBA_DROPOUT = 0.05


# --- VoxelAttentionLayer Constants ---
VOXEL_MAX_GRID_SIZE = 64
VOXEL_CA_STEPS = 5
VOXEL_DROPOUT = 0.1

# --- Linformer Block Constants (Adjusted for tiny model) ---
LINFORMER_K_PROJ = 16
LINFORMER_DFF = 64
LINFORMER_DROPOUT = 0.05
LINFORMER_FFN_DROPOUT = 0.05

# --- Adapter Block Constants ---
ADAPTER_DROPOUT = 0.1

# Assume InterleavedRoPE and all other provided layers (StackableChunkingAttentionBlock,
# DynamicVoxelAttentionLayer, LinformerBlock) are defined in the environment.

# --- Refactored base_model Construction ---

# 1. Input Layer
inp = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32)

# 2. Embedding & Initial Processing
embedded = tf.keras.layers.Embedding(
    input_dim=VOCABULARY_SIZE,
    output_dim=EMBEDDING_DIM,
    input_length=MAX_SEQ_LENGTH,
    mask_zero=False,
    name="base_embedding"
)(inp)

# iRoPE Stream
position_embedding = InterleavedRoPE(
    dim=EMBEDDING_DIM,
    max_seq_len=MAX_SEQ_LENGTH,
    name="interleaved_rope"
)(embedded)

# Skip Connection Stream is the `embedded` tensor itself

# Stream Merging: Use a GatedMergeLayer for optimal combination
initial_merge = GatedMergeLayer(d_model=EMBEDDING_DIM, name="initial_stream_merge")
x = initial_merge([embedded, position_embedding])
x = tf.keras.layers.Dropout(POSITIONAL_EMBEDDING_DROPOUT, name="initial_dropout")(x)

# 3. Core Attention/Processing Stack (Sequential Order)

# --- Block 1: SingleHeadChunkedAttention ---
x = ChunkedAttentionBlock(
    d_model=EMBEDDING_DIM,
    k_proj=K_PROJ_CHUNKED,
    dff=DFF_CHUNKED,
    dropout_rate=DROPOUT_RATE_CHUNKED,
    name="chunked_attention_block"
)(x)

# --- Block 2: MAMBA Layer ---
x = MambaBlock(
    d_model=EMBEDDING_DIM,
    d_state=MAMBA_D_STATE,
    d_conv=MAMBA_D_CONV,
    expand=MAMBA_EXPAND,
    dropout_rate=MAMBA_DROPOUT,
    name="mamba_block"
)(x)

# --- Block 3: VoxelAttentionLayer ---
x = VoxelBlock(
    d_model=EMBEDDING_DIM,
    dropout_rate=VOXEL_DROPOUT,
    max_voxel_grid_size=VOXEL_MAX_GRID_SIZE,
    ca_steps=VOXEL_CA_STEPS,
    name="voxel_block"
)(x)


# --- Block 4: Linformer Layer ---
x = LinformerBlock(
    d_model=EMBEDDING_DIM,
    k_proj=LINFORMER_K_PROJ,
    dff=LINFORMER_DFF,
    dropout_rate=LINFORMER_DROPOUT,
    ffn_dropout_rate=LINFORMER_FFN_DROPOUT,
    name="linformer_block"
)(x)


# 4. Adapter Block
# Reduces dimension from (BATCH, SEQ_LEN, EMBEDDING_DIM) to (BATCH, SEQ_LEN)
# using the new custom AdapterBlock layer.
flattened_output = AdapterBlock(
    d_model=EMBEDDING_DIM,
    dropout_rate=ADAPTER_DROPOUT,
    name="adapter_block"
)(x)


# 5. Final Model Assembly
cerebros_base_model = tf.keras.Model(
    inputs=inp,
    outputs=flattened_output, # Output shape is now (BATCH_SIZE, MAX_SEQ_LENGTH)
    name="cerebros_base_model"
)

# Display the model summary to verify the architecture
cerebros_base_model.summary()

# [cerebros_base_model] becomes the argument for `base_models`
# on CerebrosSimpleRandomSearch(base_models=[cerebros_base_model])
# and other than adding the new constants and setting them to the correct values,
# the code for the train an LLM script remain the same...
