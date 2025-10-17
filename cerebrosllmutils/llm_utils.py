"""
Utility package with LLM components.



"""


from typing import List, Tuple, Any
import tensorflow as tf



def prepare_data(
        data_0: List[str],
        tokenizer_0: Any,
        max_seq_length: int = 1024,
        prompt_length: int = 1) -> Tuple[List[List[int]], List[List[int]], int]:
    """
    Prepares tokenized input sequences and corresponding labels for training the Cerebros
    [not so] large language model.

    This function takes raw text data, tokenizes it, and applies a sliding window approach to
    generate input-label pairs for next-token prediction tasks. It assumes that each sample may
    contain a special token `</prompt>` which separates the prompt from the completion. If this
    token is not present, the sample is treated as a non-instruct example and a default prompt
    length (1 token) is used.

    For each token after the prompt (up to the first padding token), it creates an input sequence
    consisting of all tokens up to (but not including) that token, and sets the label as a one-hot
    encoded vector of the target token. A final sample is added where the label is the pad token,
    indicating the end of the sequence.

    Parameters:
    -----------
    data_0 : list of str
        List of input text samples to be processed.
    max_seq_length : int, optional: default = 1024
        Maximum sequence length for input tensors. Sequences longer than this will be truncated,
        and shorter ones will be padded. Defaults to `MAX_SEQ_LENGTH`.
    prompt_length: int, optional: Default = 1
        Rarely changed, deprecated (for R and D use), to be removed: The number of tokens fed to
        the model at training before the model is expected to start predicting the next token.
    tokenizer : a transformers.Tokenizer

    Returns:
    --------
    tuple:
        - all_input_ids (2d list of int): Tuple[List[List[int]]] Token IDs for each input sequence, shaped
          [num_samples, max_seq_length].
        - all_labels (2d list of int): Tuple[List[List[int]]] One-hot encoded labels for next-token prediction,
          shaped [num_samples, vocab_size].
        - vocab_size (int): Size of the tokenizer's vocabulary, used for label dimensions.

    Notes:
    ------
    - Special tokens like `</prompt>` are handled manually; no automatic special token insertion.
    - Padding is done using the tokenizer's pad token ID to MAX_SEQ_LENGTH.
    - The function assumes global variables `tokenizer`, `MAX_SEQ_LENGTH`, `PROMPT_LENGTH`, and
      `vocab_size` are defined in the scope where this function is called.
    """

    all_input_ids = []
    all_labels = []

    pad_token_id = tokenizer_0.pad_token_id

    # Tokenize all data at once for efficiency
    tokenized_data = tokenizer_0(
        data_0,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=False  # We'll handle special tokens manually
    )
    vocab_size = len(tokenizer_0)

    # Get the token ID for </prompt>
    end_prompt_token_id = tokenizer_0.encode("</prompt>", add_special_tokens=False)[0]

    # Process each sample
    for sample_tokens in tokenized_data['input_ids']:
        # Find the index of </prompt> token
        try:
            end_prompt_index = sample_tokens.index(end_prompt_token_id)
        except ValueError:
            # If </prompt> not found, treat sample as a non-instruct sample
            end_prompt_index = (
                        prompt_length - 1)  # int(np.ceil(len(sample_tokens) * (1/3)))  # 0 ## 1. Give it a fair starting place to predict the next word 2. reduce the number of expanded samples

        # Find first pad token after </prompt>
        first_pad_index = None
        for i in range(end_prompt_index + 1, len(sample_tokens)):
            if sample_tokens[i] == pad_token_id:
                first_pad_index = i
                break

        # If no pad token found, use the end of sequence
        if first_pad_index is None:
            first_pad_index = len(sample_tokens)

        # Apply sliding window from after </prompt> to first pad token
        # Start from end_prompt_index + 1 (first token to predict)
        # End at first_pad_index - 1 (last token to predict)
        for i in range(end_prompt_index + 1, first_pad_index):
            # Input: from start up to (but not including) token i
            input_ids = sample_tokens[:i]

            # Pad or truncate to max_seq_length
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]
            else:
                input_ids = input_ids + [pad_token_id] * (max_seq_length - len(input_ids))

            # Label: one-hot encoding of token at position i
            next_token = sample_tokens[i]
            label = [0] * vocab_size
            label[next_token] = 1

            all_input_ids.append(input_ids)
            all_labels.append(label)

        # Add final sample with pad token as label to indicate termination
        if first_pad_index < len(sample_tokens):  # Only if there's actually a pad token
            input_ids = sample_tokens[:first_pad_index]

            # Pad or truncate to max_seq_length
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]
            else:
                input_ids = input_ids + [pad_token_id] * (max_seq_length - len(input_ids))

            # Label: one-hot encoding of pad token
            label = [0] * vocab_size
            label[pad_token_id] = 1

            all_input_ids.append(input_ids)
            all_labels.append(label)

    return all_input_ids, all_labels, vocab_size


# --- Base Rotary Positional Embedding
@tf.keras.utils.register_keras_serializable()
class RotaryEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, max_seq_len=1024, temperature=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        # Ensure dim is even right at initialization
        if self.dim % 2 != 0:
            raise ValueError(f"Embedding dimension `dim` ({self.dim}) must be even for RotaryEmbedding.")
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        # *** No calculation or storage of inv_freq here or in build ***

    def build(self, input_shape):
        # Build should primarily be for creating trainable weights, which we don't have.
        # Call super().build() for Keras compatibility.
        super().build(input_shape)

    def call(self, x):  # Removed seq_len argument, calculate from x
        shape = tf.shape(x)
        batch_size = shape[0]
        actual_seq_len = shape[1]

        # *** Calculate inv_freq inside call ***
        inv_freq_base = tf.range(0, self.dim, 2, dtype=tf.float32)
        inv_freq = 1.0 / (self.temperature ** (inv_freq_base / self.dim))
        # Ensure inv_freq has the correct shape [dim/2]
        inv_freq = tf.cast(inv_freq, dtype=x.dtype)  # Match dtype early

        # Use actual_seq_len for calculations
        position = tf.range(actual_seq_len, dtype=x.dtype)  # Match dtype

        # Calculate sinusoid input using einsum or broadcasting
        # Einsum approach: Ensure correct dimensions [seq_len, dim/2]
        sinusoid_inp = tf.einsum("i,j->ij", position, inv_freq)

        # Calculate sin and cos based on the actual sequence length
        sin = tf.sin(sinusoid_inp)
        cos = tf.cos(sinusoid_inp)

        # Repeat sin/cos for interleaving: [a, b] -> [a, a, b, b]
        # Result needs shape [actual_seq_len, dim]
        sin = tf.repeat(sin, 2, axis=-1)
        cos = tf.repeat(cos, 2, axis=-1)

        # Expand dims for batch and tile
        # Output shape needs to be [batch_size, actual_seq_len, dim]
        # Add batch dimension: [1, actual_seq_len, dim]
        sin = tf.expand_dims(sin, axis=0)
        cos = tf.expand_dims(cos, axis=0)

        # Tile to match the batch size: [batch_size, actual_seq_len, dim]
        sin = tf.tile(sin, [batch_size, 1, 1])
        cos = tf.tile(cos, [batch_size, 1, 1])

        # Casting to x.dtype was already done for inv_freq, sin/cos will inherit
        # sin = tf.cast(sin, x.dtype) # Already done via calculation chain
        # cos = tf.cast(cos, x.dtype) # Already done via calculation chain

        # Return sin and cos needed by InterleavedRoPE
        return sin, cos

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "max_seq_len": self.max_seq_len,
            "temperature": self.temperature,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# iRoPE helper functions

@tf.keras.utils.register_keras_serializable()
def split_alternate(x):
    shape = tf.shape(x)
    x = tf.reshape(x, [shape[0], shape[1], shape[2] // 2, 2])
    x = tf.transpose(x, [0, 1, 3, 2])
    x = tf.reshape(x, [shape[0], shape[1], -1])
    return x


@tf.keras.utils.register_keras_serializable()
def rotate_half(x):
    x = split_alternate(x)
    d = tf.shape(x)[-1]
    rotated_x = tf.concat([-x[..., d // 2:], x[..., :d // 2]], axis=-1)
    return tf.reshape(rotated_x, tf.shape(x))


@tf.keras.utils.register_keras_serializable()
def apply_rotary_pos_emb(x, sin, cos):
    cos = tf.reshape(cos, [tf.shape(cos)[0], tf.shape(cos)[1], -1])
    sin = tf.reshape(sin, [tf.shape(sin)[0], tf.shape(sin)[1], -1])
    x_rotated = x * cos + rotate_half(x) * sin
    return x_rotated


# interleaved Rotary Postional Embedding (iRoPE)
@tf.keras.utils.register_keras_serializable()
class InterleavedRoPE(tf.keras.layers.Layer):
    def __init__(self, dim, max_seq_len=1024, **kwargs):
        super().__init__(**kwargs)
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension `dim` ({dim}) must be even for InterleavedRoPE.")
        self.dim = dim
        self.max_seq_len = max_seq_len
        # Instantiate the RotaryEmbedding layer
        # Ensure the name is consistent if needed for saving/loading
        self.rotary_emb = RotaryEmbedding(dim, max_seq_len, name="rotary_embedding")

    def call(self, x):
        # Get sin and cos from the RotaryEmbedding layer's call method
        # *** Pass only 'x'. RotaryEmbedding calculates seq_len internally. ***
        sin, cos = self.rotary_emb(x)

        # Apply the positional embeddings
        x_embedded = apply_rotary_pos_emb(x, sin, cos)
        return x_embedded

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "max_seq_len": self.max_seq_len,
        })
        # Keras handles nested layer serialization automatically
        return config

    @classmethod
    def from_config(cls, config):
        # Keras handles nested layer restoration automatically
        return cls(**config)
