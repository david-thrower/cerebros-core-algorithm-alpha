"""
Utility package with LLM components.



"""

from typing import List, Tuple, Any
import tensorflow as tf
from warnings import warn


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
@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='RotaryEmbedding')
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

@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='split_alternate')
def split_alternate(x):
    shape = tf.shape(x)
    x = tf.reshape(x, [shape[0], shape[1], shape[2] // 2, 2])
    x = tf.transpose(x, [0, 1, 3, 2])
    x = tf.reshape(x, [shape[0], shape[1], -1])
    return x


@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='rotate_half')
def rotate_half(x):
    x = split_alternate(x)
    d = tf.shape(x)[-1]
    rotated_x = tf.concat([-x[..., d // 2:], x[..., :d // 2]], axis=-1)
    return tf.reshape(rotated_x, tf.shape(x))


@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='apply_rotary_pos_emb')
def apply_rotary_pos_emb(x, sin, cos):
    cos = tf.reshape(cos, [tf.shape(cos)[0], tf.shape(cos)[1], -1])
    sin = tf.reshape(sin, [tf.shape(sin)[0], tf.shape(sin)[1], -1])
    x_rotated = x * cos + rotate_half(x) * sin
    return x_rotated


# interleaved Rotary Postional Embedding (iRoPE)
@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='InterleavedRoPE')
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


@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='Perplexity')
class Perplexity(tf.keras.metrics.Metric):
    """
    Computes perplexity, defined as e^(categorical crossentropy).
    """

    def __init__(self, name='perplexity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_crossentropy = self.add_weight(name='total_crossentropy', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate categorical crossentropy
        crossentropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # Update the running sum of crossentropy and the count of samples
        self.total_crossentropy.assign_add(tf.reduce_sum(crossentropy))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], dtype=tf.float32))

    def result(self):
        # Compute the average crossentropy
        average_crossentropy = self.total_crossentropy / self.count
        # Compute perplexity as e^(average crossentropy)
        return tf.exp(average_crossentropy)

    def reset_state(self):
        # Reset the state variables
        self.total_crossentropy.assign(0.0)
        self.count.assign(0.0)


@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='CerebrosNotGPTConfig')
class CerebrosNotGPTConfig:
    def __init__(self, max_sequence_length=1536, padding_token=None):
        self.max_sequence_length = max_sequence_length
        self.padding_token = padding_token

    def get_config(self):
        return {
            'max_sequence_length': self.max_sequence_length,
            'padding_token': self.padding_token
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='CerebrosNotGPT')
class CerebrosNotGPT(tf.keras.Model):
    def __init__(self, config: Any, model: Any = None, **kwargs):
        # 1. Store the nested model argument.
        self.config = config
        self.model = model
        
        # 2. Extract and remove custom kwargs (like 'model') before calling super.
        #    This is important to prevent 'unrecognized keyword argument' errors.
        #    The nested model is already extracted and stored, so it can be safely removed.
        kwargs.pop('model', None)
        
        # 3. Call the parent constructor with the cleaned kwargs.
        super().__init__(**kwargs)

        self.max_sequence_length = config.max_sequence_length
        self.padding_token = config.padding_token

    def get_config(self):
        base_config = super().get_config()
        config_dict = {
            'config': self.config.get_config(),
        }
        
        # Explicitly handle nested model serialization.
        # This is required if Keras's automatic tracking fails.
        if self.model is not None:
            # Note: This approach might still suffer from weight loss.
            # The recommended way is to let Keras handle it automatically.
            config_dict['model'] = tf.keras.utils.serialize_keras_object(self.model)

        base_config.update(config_dict)
        return base_config

    @classmethod
    def from_config(cls, config):
        # Separate the custom config.
        config_obj_dict = config.pop('config')
        config_obj = CerebrosNotGPTConfig.from_config(config_obj_dict)
        
        # Manually extract and load the nested model.
        nested_model_config = config.pop('model', None)
        if nested_model_config:
            nested_model = tf.keras.utils.deserialize_keras_object(nested_model_config)
        else:
            nested_model = None
            
        # Reconstruct the outer model by passing the restored parts.
        return cls(config=config_obj, model=nested_model, **config)

    def call(self, inputs, training=False):
        if self.model is None:
            raise ValueError("Inner model not initialized properly")
        return self.model(inputs, training=training)

    @staticmethod
    def apply_top_k_probs(probs, k):
        if k is None or k <= 0:
            return probs
        # Flatten and argsort for indices
        sorted_indices = tf.argsort(probs, direction='DESCENDING')
        keep_indices = sorted_indices[:k]
        mask = tf.zeros_like(probs, dtype=tf.bool)
        mask = tf.tensor_scatter_nd_update(mask, tf.reshape(keep_indices, (-1, 1)),
                                           tf.ones((k,), dtype=tf.bool))
        filtered_probs = tf.where(mask, probs, tf.zeros_like(probs))
        # Renormalize
        filtered_probs = filtered_probs / tf.reduce_sum(filtered_probs)
        return filtered_probs

    @staticmethod
    def apply_top_p_probs(probs, p):
        if p is None or p >= 1.0:
            return probs
        sorted_indices = tf.argsort(probs, direction='DESCENDING')
        sorted_probs = tf.gather(probs, sorted_indices)
        cumulative_probs = tf.cumsum(sorted_probs)
        mask = cumulative_probs <= p
        # Always keep at least 1 token
        mask = tf.concat([tf.constant([True]), mask[1:]], axis=0)
        keep_indices = tf.boolean_mask(sorted_indices, mask)
        filtered_probs = tf.where(
            tf.reduce_any(tf.equal(tf.range(tf.shape(probs)[0])[:, None], keep_indices), axis=1), probs,
            tf.zeros_like(probs))
        # Renormalize
        filtered_probs = filtered_probs / tf.reduce_sum(filtered_probs)
        return filtered_probs

    def generate(self,
                 token_ids,
                 do_sample=False,
                 max_new_tokens=None,
                 temperature=1.0,
                 top_k=None,
                 top_p=None,
                 frequency_penalty=None,
                 presence_penalty=None,
                 repetition_penalty=None):
        """
        Generate text autoregressively from token IDs.
        Applies filtering in sequence: penalties -> temperature -> top-k -> top-p
        """
        # Convert token_ids to list if it's not already
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)

        # Determine the actual maximum number of new tokens
        if max_new_tokens is None:
            max_new_tokens = self.max_sequence_length - len(token_ids)
        else:
            max_new_tokens = min(max_new_tokens, self.max_sequence_length - len(token_ids))

        # Initialize the generated tokens list
        generated_tokens = []
        current_tokens = token_ids.copy()

        # Autoregressive generation loop
        for _ in range(max_new_tokens):
            # Pad or truncate to max_sequence_length
            if len(current_tokens) > self.max_sequence_length:
                input_tokens = current_tokens[-self.max_sequence_length:]
            else:
                padding_needed = self.max_sequence_length - len(current_tokens)
                input_tokens = current_tokens + [self.padding_token] * padding_needed

            # Convert to tensor and get model prediction
            input_tensor = tf.constant([input_tokens], dtype=tf.int32)
            probs_nested = self.model(input_tensor)
            probs = probs_nested[0]  # Already softmax probabilities (NOT logits as comment says)
            logits = tf.math.log(probs + 10 ** -20)  # Convert to logits for penalty application

            if do_sample:
                # Apply repetition/frequency/presence penalties to logits
                if frequency_penalty is not None or presence_penalty is not None:
                    # Collect token counts from current_tokens
                    token_counts = {}
                    for t in current_tokens:
                        token_counts[t] = token_counts.get(t, 0) + 1

                    # Prepare penalty tensor
                    vocab_size = tf.shape(logits)[0]
                    penalties = tf.zeros_like(logits)

                    for token_id, count in token_counts.items():
                        if token_id >= vocab_size:
                            continue
                        penalty = 0.0
                        if presence_penalty is not None:
                            penalty += presence_penalty
                        if frequency_penalty is not None:
                            penalty += frequency_penalty * count

                        penalties = tf.tensor_scatter_nd_add(
                            penalties,
                            [[token_id]],
                            [penalty]
                        )

                    # Subtract penalties from logits
                    logits = logits - penalties

                # Apply repetition penalty (standard approach)
                if repetition_penalty is not None and repetition_penalty != 1.0:
                    # Collect unique tokens that have appeared
                    unique_tokens = list(set(current_tokens))
                    vocab_size = tf.shape(logits)[0]

                    for token_id in unique_tokens:
                        if token_id < vocab_size:
                            # Divide logits of repeated tokens by penalty
                            logits = tf.tensor_scatter_nd_update(
                                logits,
                                [[token_id]],
                                [logits[token_id] / repetition_penalty]
                            )

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Convert to probabilities
                probs = tf.nn.softmax(logits)

                # Apply top-k filtering (if specified)
                if top_k is not None and top_k > 0:
                    k = min(top_k, tf.shape(probs)[0])
                    # Get top-k values and indices
                    top_k_values, top_k_indices = tf.nn.top_k(probs, k=k, sorted=False)
                    # Create mask for top-k positions
                    top_k_mask = tf.scatter_nd(
                        tf.expand_dims(top_k_indices, 1),
                        tf.ones_like(top_k_values, dtype=tf.bool),
                        tf.shape(probs)
                    )
                    # Zero out non-top-k probabilities
                    probs = tf.where(top_k_mask, probs, tf.zeros_like(probs))
                    # Renormalize
                    probs = probs / tf.reduce_sum(probs)
                    print(
                        f">>> After top_k: {tf.shape(probs)} shape, {tf.reduce_sum(tf.cast(probs > 1e-8, tf.int32))} non-zero probs")

                # Apply top-p filtering (if specified)
                if top_p is not None and top_p < 1.0:
                    # Sort probabilities in descending order
                    sorted_indices = tf.argsort(probs, direction='DESCENDING')
                    sorted_probs = tf.gather(probs, sorted_indices)
                    cumulative_probs = tf.cumsum(sorted_probs)
                    # Create mask for top-p
                    mask = cumulative_probs <= top_p
                    # Always keep at least one token
                    mask = tf.concat([tf.constant([True]), mask[1:]], axis=0)
                    # Get indices to keep
                    keep_indices = tf.boolean_mask(sorted_indices, mask)
                    # Create mask for original indices
                    filter_mask = tf.scatter_nd(
                        tf.expand_dims(keep_indices, 1),
                        tf.ones_like(keep_indices, dtype=tf.bool),
                        tf.shape(probs)
                    )
                    # Apply mask and renormalize
                    probs = tf.where(filter_mask, probs, tf.zeros_like(probs))
                    probs = probs / tf.reduce_sum(probs)
                    print(
                        f">>> After top_p: {tf.shape(probs)} shape, {tf.reduce_sum(tf.cast(probs > 1e-8, tf.int32))} non-zero probs")

                # Sample from the final filtered distribution
                # Get non-zero indices and their probabilities
                non_zero_mask = probs > 1e-8
                if tf.reduce_any(non_zero_mask):
                    filtered_indices = tf.where(non_zero_mask)[:, 0]  # Get indices
                    filtered_probs = tf.boolean_mask(probs, non_zero_mask)  # Get probabilities
                    # Sample
                    sampled_local_index = tf.random.categorical(tf.math.log(filtered_probs)[None, :], 1)[0, 0]
                    # Map back to vocabulary index
                    next_token_id = int(filtered_indices[sampled_local_index].numpy())
                else:
                    # Fallback if all probabilities are zero
                    warn(
                        "Token sampling had to revert to greedy sampling, because no probs had a value > 0, unexpected")
                    next_token_id = int(tf.argmax(probs, axis=-1).numpy())

            else:
                # Greedy sampling (argmax) - apply repetition penalty if needed
                if repetition_penalty is not None and repetition_penalty != 1.0:
                    unique_tokens = list(set(current_tokens))
                    vocab_size = tf.shape(logits)[0]
                    for token_id in unique_tokens:
                        if token_id < vocab_size:
                            logits = tf.tensor_scatter_nd_update(
                                logits,
                                [[token_id]],
                                [logits[token_id] / repetition_penalty]
                            )

                next_token_id = int(tf.argmax(logits, axis=-1).numpy())

            # Check for termination condition
            if next_token_id == self.padding_token:
                break

            # Add to generated tokens and update current tokens
            generated_tokens.append(int(next_token_id))
            current_tokens.append(int(next_token_id))

            # Check if we've reached max sequence length
            if len(current_tokens) >= self.max_sequence_length:
                break

        return token_ids + generated_tokens

# A custom schedule: Cosine decay with some warm - up steps
@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='WarmupCosineDecayRestarts')
class WarmupCosineDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    A learning rate schedule that combines a linear warmup with cosine decay restarts.
    This version is compatible with TensorFlow's graph execution (used in model.fit).
    """

    def __init__(self, initial_learning_rate, warmup_steps, first_decay_steps, t_mul=2.0, m_mul=1.0, alpha=0.0):
        super().__init__()

        # Store all parameters as public attributes for get_config serialization
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha

        # Create the CosineDecayRestarts schedule for internal logic.
        # The parameters passed here are the same ones we just stored.
        self.cosine_restarts_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha
        )


    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)

        # Calculate the learning rate for both phases unconditionally
        warmup_lr = self.initial_learning_rate * step / self.warmup_steps

        # The cosine schedule is designed to start from step 0, so we give it
        # the "post-warmup" step count.
        decay_lr = self.cosine_restarts_schedule(step - self.warmup_steps)

        # Create a multiplier that is 1.0 during warmup and 0.0 after.
        # tf.cast(condition, tf.float32) converts a boolean tensor to 1.0 or 0.0.
        warmup_multiplier = tf.cast(step < self.warmup_steps, tf.float32)

        # The decay multiplier is the opposite.
        decay_multiplier = 1.0 - warmup_multiplier

        # Combine the two learning rates. Only one will be active at a time.
        return (warmup_multiplier * warmup_lr) + (decay_multiplier * decay_lr)

    def get_config(self):
        # Use the stored public attributes for the config.
        # This bypasses the issue of accessing private attributes (_t_mul) from
        # the nested Keras object, which can be brittle.
        config = {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self.t_mul,
            "m_mul": self.m_mul,
            "alpha": self.alpha,
        }

        # Use from_config to properly allow deserialization
        return config
