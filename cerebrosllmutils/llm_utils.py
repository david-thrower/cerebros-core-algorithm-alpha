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
        prompt_length: int = 1) -> Tuple[List[List[int]], List[int], int]:
    """
    Prepares tokenized input sequences and integer labels for next-token prediction.

    This function tokenizes input texts and applies a sliding window to generate
    (input, label) pairs. For each token in a sequence, the input consists of all
    preceding tokens, and the label is the integer ID of the current token.

    The sliding window for each sample stops after it has generated a label that
    matches the tokenizer's ``pad_token_id``. This teaches the model to predict
    this specific token to signal the end of a generated sequence.

    Parameters
    ----------
    data_0 : List[str]
        Raw text samples to be processed.
    tokenizer_0 : Any
        A ``transformers``-style tokenizer (must provide ``pad_token_id`` and
        ``encode``).
    max_seq_length : int, default 1024
        Length to which all input sequences are padded or truncated.
    prompt_length : int, default 1
        Number of tokens treated as the prompt when the special ``</prompt>``
        token is absent.

    Returns
    -------
    Tuple[List[List[int]], List[int], int]
        * ``all_input_ids`` – list of padded input sequences, shape
          ``(NUM_EXPANDED_SAMPLES, max_seq_length)``.
        * ``all_labels`` – list of integer token IDs for the next token,
          shape ``(NUM_EXPANDED_SAMPLES,)``.
        * ``vocab_size`` – size of the tokenizer vocabulary.

    Notes
    -----
    * The tokenizer's ``pad_token_id`` is used for two purposes:
      1. To pad the input sequences to ``max_seq_length``.
      2. To act as the special label that terminates the sliding window for a
         sample, teaching the model when to stop generating text.
    * Labels are returned as plain integers, not one-hot encoded vectors.
    """
    all_input_ids: List[List[int]] = []
    all_labels: List[int] = []

    # This token is used for both sequence padding and as the termination label.
    pad_token_id = tokenizer_0.pad_token_id

    # Tokenize all data at once for efficiency
    tokenized_data = tokenizer_0(
        data_0,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=False
    )
    vocab_size = len(tokenizer_0)

    # Get the token ID for </prompt>
    try:
        end_prompt_token_id = tokenizer_0.encode("</prompt>", add_special_tokens=False)[0]
    except IndexError:
        warn("Tokenizer does not seem to have a token for '</prompt>'. "
             "Function will rely on prompt_length for all samples.")
        # Set to a value that will not be found in the sequence
        end_prompt_token_id = -1

    # Process each sample
    for sample_tokens in tokenized_data['input_ids']:
        # Find the index of </prompt> token
        try:
            end_prompt_index = sample_tokens.index(end_prompt_token_id)
        except ValueError:
            # If </prompt> not found, treat sample as a non-instruct sample
            end_prompt_index = prompt_length - 1

        # Apply sliding window from after the prompt to the end of the sequence.
        # The loop will break when the label is the pad token.
        for i in range(end_prompt_index + 1, len(sample_tokens)):
            # Input: tokens up to (but not including) position i
            input_ids = sample_tokens[:i]

            # Pad or truncate to max_seq_length using the pad token
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]
            else:
                input_ids = input_ids + [pad_token_id] * (max_seq_length - len(input_ids))

            # Label: integer token ID at position i
            label = sample_tokens[i]

            all_input_ids.append(input_ids)
            all_labels.append(label)

            # Stop after the first occurrence of the pad token label
            if label == pad_token_id:
                break

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


@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='SparsePerplexity')
class SparsePerplexity(tf.keras.metrics.Metric):
    """
    Computes perplexity for a batch of next-token predictions.
    
    Expects:
        y_true: (Batch_Size,) - Integer labels (the actual next token).
        y_pred: (Batch_Size, Vocab_Size) - Logits/Probabilities for the next token.
    """

    def __init__(self, name='perplexity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_crossentropy = self.add_weight(name='total_crossentropy', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true shape: (Batch_Size,)
        # y_pred shape: (Batch_Size, Vocab_Size)
        
        # Calculate sparse categorical crossentropy
        # from_logits=True is safer for raw model outputs. 
        # If your final layer is Softmax, change to False.
        crossentropy = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, 
            y_pred, 
            from_logits=True
        )
        
        # Handle sample weighting
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            crossentropy = crossentropy * sample_weight
            batch_weight_sum = tf.reduce_sum(sample_weight)
        else:
            # Count is the Batch Size
            batch_weight_sum = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)

        # Update the running sum of crossentropy
        self.total_crossentropy.assign_add(tf.reduce_sum(crossentropy))
        
        # Update the running count
        self.count.assign_add(batch_weight_sum)

    def result(self):
        # Compute the average crossentropy
        average_crossentropy = tf.math.divide_no_nan(self.total_crossentropy, self.count)
        
        # Compute perplexity as e^(average crossentropy)
        return tf.exp(average_crossentropy)

    def reset_state(self):
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


# Gating merge layer

@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='GatedMergeLayer')
class GatedMergeLayer(tf.keras.layers.Layer):
    """
    Merges two input streams using a learned gating mechanism.

    The gate is computed from the first input stream and determines the
    proportion of each stream in the final output.
    output = gate * input_1 + (1 - gate) * input_2

    Args:
        d_model (int): The feature dimension of the input streams.
    """
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        # A dense layer to generate the gate values (between 0 and 1)
        self.gate_dense = tf.keras.layers.Dense(d_model, activation='sigmoid')

    def call(self, inputs):
        input_1, input_2 = inputs
        # Generate gate from the first input
        gate_values = self.gate_dense(input_1)
        # Blend the two streams
        return gate_values * input_1 + (1.0 - gate_values) * input_2

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
        })
        return config


## Attention Block 1: Chunked Attention (Big - Bird - Like)
# Captures short and mid range token to token relationships
# effectively. Is very computationally efficient.

@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='SingleHeadChunkedAttentionSameDimOutput')
class SingleHeadChunkedAttentionSameDimOutput(tf.keras.layers.Layer):
    """
    A single-head attention mechanism with a chunked compression and an output
    of the same dimensionality as the input.

    This layer is designed to produce an output tensor of shape
    (batch_size, sequence_length, d_model) for each token in the input sequence.
    It uses the "Chunked Attention with Context" method to efficiently compress
    Keys and Values, making it suitable for long sequences.

    Args:
        d_model (int): The dimension of the input embeddings (e.g., 512, 768).
        k_proj (int): The target sequence length after chunking.
                      The original sequence length must be a multiple of this.
    """
    def __init__(self, d_model, k_proj, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.k_proj = k_proj
        self.compressed_dim = 2 * d_model

        # Standard linear projections to get Q, K, and V from the input
        self.q_dense = tf.keras.layers.Dense(d_model)
        self.k_dense = tf.keras.layers.Dense(d_model)
        self.v_dense = tf.keras.layers.Dense(d_model)

        # Project Q to match the new compressed dimension of K and V
        self.q_compression_dense = tf.keras.layers.Dense(self.compressed_dim)

        # A small, learned layer to create the context vector from a chunk summary
        self.summary_context_dense = tf.keras.layers.Dense(d_model)

        # === FINAL OUTPUT PROJECTION ===
        # To project the context vector back to the original embedding dimension,
        # we use a small MLP. This allows for non-linear feature interactions.
        # The final Dense(d_model) layer maps the features back to the
        # original d_model space, applied independently to each token.
        self.output_mlp_1 = tf.keras.layers.Dense(self.compressed_dim, activation='relu')
        self.output_mlp_2 = tf.keras.layers.Dense(self.d_model, activation='relu')

    def build(self, input_shape):
        seq_len = input_shape[-2]
        if seq_len % self.k_proj != 0:
            raise ValueError(
                f"Sequence length ({seq_len}) must be divisible by k_proj ({self.k_proj}) "
                "for this chunked compression strategy."
            )
        super().build(input_shape)

    def _compress_kv(self, kv_tensor):
        """
        Helper function to compress Key or Value tensors using the chunked approach.
        """
        # kv_tensor shape: (BATCH_SIZE, SEQUENCE_LENGTH, D_MODEL)
        batch_size = tf.shape(kv_tensor)[0]
        seq_len = tf.shape(kv_tensor)[1]
        chunk_size = seq_len // self.k_proj

        # === Step 2a: Chunk the tensor ===
        # Shape: (BATCH_SIZE, K_PROJ, CHUNK_SIZE, D_MODEL)
        kv_reshaped = tf.reshape(kv_tensor, [batch_size, self.k_proj, chunk_size, self.d_model])

        # === Step 2b: Compute the fixed summary (mean) for each chunk ===
        # Shape: (BATCH_SIZE, K_PROJ, D_MODEL)
        summary = tf.reduce_mean(kv_reshaped, axis=2)

        # === Step 2c: Compute the learned context vector for each chunk ===
        # Shape: (BATCH_SIZE, K_PROJ, D_MODEL)
        context = self.summary_context_dense(summary)

        # === Step 2d: Concatenate summary and context ===
        # Shape: (BATCH_SIZE, K_PROJ, 2 * D_MODEL)
        kv_compressed = tf.concat([summary, context], axis=-1)

        return kv_compressed

    def call(self, inputs):
        # inputs shape: (BATCH_SIZE, SEQUENCE_LENGTH, D_MODEL)
        batch_size = tf.shape(inputs)[0]

        # === Step 1: Create Query, Key, and Value matrices ===
        # Shape: (BATCH_SIZE, SEQUENCE_LENGTH, D_MODEL)
        q = self.q_dense(inputs)
        k = self.k_dense(inputs)
        v = self.v_dense(inputs)

        # === Step 2: Compress K and V ===
        # k_compressed/v_compressed shape: (BATCH_SIZE, K_PROJ, 2 * D_MODEL)
        k_compressed = self._compress_kv(k)
        v_compressed = self._compress_kv(v)

        # === Step 3: Prepare Q for attention ===
        # q shape: (BATCH_SIZE, SEQUENCE_LENGTH, 2 * D_MODEL)
        q = self.q_compression_dense(q)

        # === Step 4: Scaled Dot-Product Attention ===
        # scores shape: (BATCH_SIZE, SEQUENCE_LENGTH, K_PROJ)
        scores = tf.matmul(q, k_compressed, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.compressed_dim, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # context_vector shape: (BATCH_SIZE, SEQUENCE_LENGTH, 2 * D_MODEL)
        context_vector = tf.matmul(attention_weights, v_compressed)

        # === Step 5: Final Output Projection to d_model using an MLP ===
        # This MLP processes each token's context vector independently.
        # Shape: (BATCH_SIZE, SEQUENCE_LENGTH, 2 * D_MODEL) -> (BATCH_SIZE, SEQUENCE_LENGTH, 2 * D_MODEL)
        x = self.output_mlp_1(context_vector)
        # Shape: (BATCH_SIZE, SEQUENCE_LENGTH, 2 * D_MODEL) -> (BATCH_SIZE, SEQUENCE_LENGTH, D_MODEL)
        output = self.output_mlp_2(x)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "k_proj": self.k_proj,
        })
        return config


# Block object that adds proper layer normalization, dropout, merging of inputs and outputs, ...
@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='ChunkedAttentionBlock')
class ChunkedAttentionBlock(tf.keras.layers.Layer):
    """
    A Transformer Block using Pre-Layer Normalization and the
    SingleHeadChunkedAttentionSameDimOutput layer.
    """
    def __init__(self, d_model, k_proj, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.k_proj = k_proj
        self.dff = dff
        self.dropout_rate = dropout_rate

        # --- Attention Sub-layer ---
        self.attention = SingleHeadChunkedAttentionSameDimOutput(
            d_model=d_model,
            k_proj=k_proj,
            name="chunked_attention"
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # --- Stream Merging Layer (GATING) ---
        # This layer generates a gate to control the flow of information
        # between the original input and the attention output.
        self.gate = GatedMergeLayer(d_model)

        # --- Feed-Forward Network (FFN) Sub-layer ---
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch, seq_len, d_model)
        ], name="feed_forward_network")
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        # --- Attention Sub-layer with Pre-LN and Gated Stream Merging ---
        # 1. Normalize inputs
        norm_x = self.layernorm1(inputs)
        # 2. Apply attention
        attn_output = self.attention(norm_x)
        # 3. Apply dropout
        attn_output = self.dropout1(attn_output, training=training)

        # 4. GATE the original input and the attention output
        # Generate the gate from the normalized input
        merged_output = self.gate([inputs, attn_output])

        # --- Feed-Forward Sub-layer with Pre-LN and Residual ---
        # 1. Normalize the output of the merged stream
        norm_merged = self.layernorm2(merged_output)
        # 2. Apply FFN
        ffn_output = self.ffn(norm_merged)
        # 3. Apply dropout
        ffn_output = self.dropout2(ffn_output, training=training)
        # 4. Residual connection
        final_output = merged_output + ffn_output

        return final_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "k_proj": self.k_proj,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        })
        return config


#### Block 2 Mamba ############
# Also effective at short range to mid - range token - to - token relationships
@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='MambaBlock')
class MambaBlock(tf.keras.layers.Layer):
    """
    A Mamba Block with Pre-Layer Normalization and a Gated Residual Connection.

    This block implements a simplified Selective State Space Model, designed for
    linear sequence modeling. It includes a 1D convolution for local context
    and a selective scan mechanism for long-range dependencies.

    Args:
        d_model (int): The dimension of the input embeddings.
        d_state (int): The dimension of the latent state (B).
        d_conv (int): The kernel size of the 1D convolution.
        expand (int): The expansion factor for the inner projection dimension.
        dropout_rate (float): Dropout rate for the block's output.
    """

    def __init__(self, d_model, d_state, d_conv, expand, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout_rate = dropout_rate

        self.d_inner = int(self.expand * self.d_model)

        # --- Normalization and Dropout ---
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # --- Core Mamba Components ---
        # Input projection
        self.in_proj = tf.keras.layers.Dense(self.d_inner * 2, use_bias=False)

        # 1D Convolution for local processing
        # --- CORRECTION: The filters and groups should be d_inner, not d_inner * 2 ---
        # This is because the convolution operates on the tensor AFTER the GLU split,
        # which has d_inner channels.
        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.d_inner,
            kernel_size=self.d_conv,
            padding='causal',
            groups=self.d_inner,
            use_bias=False,
            activation='silu'
        )

        # Selective SSM parameters (A, B, C, D)
        # A is a fixed matrix, B and C are data-dependent (selective)
        self.A_log = self.add_weight(
            shape=(self.d_inner, self.d_state),
            initializer='random_normal',
            trainable=True,
            name='A_log'
        )
        self.D = self.add_weight(
            shape=(self.d_inner,),
            initializer='ones',
            trainable=True,
            name='D'
        )
        # Projects x to get the time-step delta (dt)
        self.dt_proj = tf.keras.layers.Dense(self.d_inner, use_bias=True)

        # Projects x to get B and C
        self.x_proj = tf.keras.layers.Dense(self.d_state * 2, use_bias=False)

        # Output projection
        self.out_proj = tf.keras.layers.Dense(self.d_model, use_bias=False)

        # --- Gated Merge for Residual Connection ---
        self.gated_merge = GatedMergeLayer(d_model)

    def build(self, input_shape):
        # Adding a build method to silence the UserWarning and follow best practices.
        # All sub-layers (Dense, Conv1D, etc.) are built automatically by Keras,
        # so we just need to call the superclass's build method.
        super().build(input_shape)

    def _selective_scan(self, u, delta, A, B, C, D):
        """
        Vectorized selective scan operation.
        Args:
            u: (batch, len, d_inner)
            delta: (batch, len, d_inner)
            A: (d_inner, d_state)
            B: (batch, len, d_state)
            C: (batch, len, d_state)
            D: (d_inner,)
        Returns:
            y: (batch, len, d_inner)
        """
        batch_size = tf.shape(u)[0]
        seq_len = tf.shape(u)[1]

        # Discretize A and B
        # dA shape: (batch, len, d_inner, d_state)
        dA = tf.exp(tf.einsum('bld,dn->bldn', delta, A))
        # dB shape: (batch, len, d_inner, d_state)
        dB = tf.einsum('bld,bln->bldn', delta, B)

        # --- CORRECTION: Create a matching initializer tuple ---
        # Initial state
        # h shape: (batch, d_inner, d_state)
        h_initial = tf.zeros((batch_size, self.d_inner, self.d_state), dtype=u.dtype)

        # Initial output
        # y_initial shape: (batch, d_inner)
        y_initial = tf.zeros((batch_size, self.d_inner), dtype=u.dtype)

        def scan_fn(prev_state, current_inputs):
            # prev_state is now a tuple: (prev_h, prev_y)
            prev_h, _ = prev_state
            # current_inputs: tuple of (u_i, dB_i, dA_i, C_i)
            u_i, dB_i, dA_i, C_i = current_inputs

            # Update state: h_t = dA_t * h_{t-1} + dB_t * u_t
            h_t = dA_i * prev_h + dB_i * u_i[:, :, tf.newaxis]

            # Calculate output: y_t = C_t^T * h_t
            # y_t shape: (batch, d_inner)
            y_t = tf.einsum('bdn,bn->bd', h_t, C_i)

            # Return a tuple to match the new initializer structure
            return h_t, y_t

        # Prepare inputs for tf.scan
        # u shape: (batch, len, d_inner) -> (len, batch, d_inner)
        scan_u = tf.transpose(u, [1, 0, 2])
        # dB shape: (batch, len, d_inner, d_state) -> (len, batch, d_inner, d_state)
        scan_dB = tf.transpose(dB, [1, 0, 2, 3])
        # dA shape: (batch, len, d_inner, d_state) -> (len, batch, d_inner, d_state)
        scan_dA = tf.transpose(dA, [1, 0, 2, 3])
        # C shape: (batch, len, d_state) -> (len, batch, d_state)
        scan_C = tf.transpose(C, [1, 0, 2])

        # Run the scan with the corrected initializer
        # y shape: (len, batch, d_inner)
        _, y = tf.scan(
            fn=scan_fn,
            elems=(scan_u, scan_dB, scan_dA, scan_C),
            initializer=(h_initial, y_initial)  # <-- Pass the tuple here
        )

        # Transpose back to (batch, len, d_inner)
        y = tf.transpose(y, [1, 0, 2])

        # Add skip connection D * u
        y = y + u * D

        return y

    def call(self, inputs, training=False):
        # --- Pre-LN and Residual Connection Setup ---
        residual = inputs
        x = self.layernorm(inputs)

        # --- Input Projection and Convolution ---
        # x shape: (batch, seq_len, d_inner * 2)
        x = self.in_proj(x)

        # Apply SiLU activation and GLU gating
        x, gate = tf.split(x, num_or_size_splits=2, axis=-1)
        x = tf.nn.silu(x) * gate

        # Apply 1D convolution
        x = self.conv1d(x)

        # --- Simplified Selective Scan ---
        # Project x to get dt, B, C
        dt = self.dt_proj(x)  # (batch, len, d_inner)
        B_C = self.x_proj(x)  # (batch, len, d_state * 2)
        B, C = tf.split(B_C, num_or_size_splits=2, axis=-1)  # (batch, len, d_state) each

        # A is a fixed matrix, we use exp(A_log) for stability
        A = -tf.exp(tf.cast(self.A_log, x.dtype))

        # Run the selective scan
        y = self._selective_scan(u=x, delta=dt, A=A, B=B, C=C, D=self.D)

        # --- Output Projection and Dropout ---
        output = self.out_proj(y)
        output = self.dropout(output, training=training)

        # --- Gated Residual Connection ---
        return self.gated_merge([residual, output])

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "dropout_rate": self.dropout_rate,
        })
        return config


# Block 3: Cellular Automata - Voxel - Simulation Attention - Mimetic 
# A wild card: Captures a full range of token - to - token relationships.
# Strategically placed after layer that encode short - range relationships
# well, so as to make the gradient landscape favor this focus on longer 
# range / higher order relationships, since it is deeper in the stack 
# of layers.
@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='DynamicVoxelAttentionLayer')
class DynamicVoxelAttentionLayer(tf.keras.layers.Layer):
    def __init__(self,
                 d_model,
                 max_voxel_grid_size=64,
                 ca_steps=5,
                 ca_kernel_size=(3, 3, 3),
                 kernel_initializer='glorot_uniform',
                 gate_locked=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_voxel_grid_size = max_voxel_grid_size
        self.ca_steps = ca_steps
        self.ca_kernel_size = ca_kernel_size
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.gate_locked = gate_locked

        # Gating weights for Q, K, V
        self.gate_k = self.add_weight(name='gate_k', shape=(self.d_model,), initializer='zeros', trainable=True)
        self.gate_q = self.add_weight(name='gate_q', shape=(self.d_model,), initializer='zeros', trainable=True)
        self.gate_v = self.add_weight(name='gate_v', shape=(self.d_model,), initializer='zeros', trainable=True)

        # Dense projections
        self.dense_k = tf.keras.layers.Dense(self.d_model, kernel_initializer=self.kernel_initializer)
        self.dense_q = tf.keras.layers.Dense(self.d_model, kernel_initializer=self.kernel_initializer)
        self.dense_v = tf.keras.layers.Dense(self.d_model, kernel_initializer=self.kernel_initializer)

        # CA components - separate convolutions for different roles
        self.ca_qk_conv = tf.keras.layers.Conv3D(
            filters=self.d_model,
            kernel_size=self.ca_kernel_size,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            name='ca_qk_conv'
        )
        self.ca_v_conv = tf.keras.layers.Conv3D(
            filters=self.d_model,
            kernel_size=self.ca_kernel_size,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            name='ca_v_conv'
        )
        self.ca_attention_conv = tf.keras.layers.Conv3D(
            filters=self.d_model,
            kernel_size=self.ca_kernel_size,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            name='ca_attention_conv'
        )
        
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        super().build(input_shape)

    def _run_ca_attention(self, q_voxel, k_voxel, v_voxel):
        """
        Run CA simulation that approximates attention without explicit matmul.
        The CA dynamics themselves compute the attention-like interactions.
        """
        # Combine Q and K to create attention-like interactions
        qk_interaction = q_voxel * k_voxel  # Element-wise interaction in voxel space
        
        # Normalize the interaction
        qk_normalized = self.layer_norm(qk_interaction)
        
        # Use CA to propagate attention information
        attention_voxel = qk_normalized * v_voxel
        
        result = attention_voxel
        for _ in range(self.ca_steps):
            # QK interaction update
            qk_update = self.ca_qk_conv(result)
            
            # V update influenced by QK
            v_update = self.ca_v_conv(result)
            
            # Combined attention update
            attention_update = self.ca_attention_conv(result)
            
            # Apply tanh nonlinearity and residual connection
            result = tf.tanh(result + qk_update + v_update + attention_update)
            
        return result

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # 1. Project inputs and apply gating for Q, K, and V
        q = self.dense_q(inputs) * tf.nn.sigmoid(self.gate_q)
        k = self.dense_k(inputs) * tf.nn.sigmoid(self.gate_k)
        v = self.dense_v(inputs) * tf.nn.sigmoid(self.gate_v)

        # 2. Reshape all three to 3D grid (sequence as depth)
        q_voxel_3d = tf.reshape(q, [batch_size, seq_len, 1, 1, self.d_model])
        k_voxel_3d = tf.reshape(k, [batch_size, seq_len, 1, 1, self.d_model])
        v_voxel_3d = tf.reshape(v, [batch_size, seq_len, 1, 1, self.d_model])

        # 3. Pad scratch space for CA simulation
        paddings = [[0, 0], [0, 0],
                    [0, self.max_voxel_grid_size - 1],
                    [0, self.max_voxel_grid_size - 1],
                    [0, 0]]
        
        q_padded = tf.pad(q_voxel_3d, paddings)
        k_padded = tf.pad(k_voxel_3d, paddings)
        v_padded = tf.pad(v_voxel_3d, paddings)

        # 4. Run the Cellular Automata simulation that approximates attention
        # This is where the attention is compressed into CA dynamics
        attention_voxel = self._run_ca_attention(q_padded, k_padded, v_padded)

        # 5. Collapse the 3D voxels back to 2D sequences
        # The reduce_mean merges the spatial dimensions, returning to (batch, seq, d_model)
        attention_output = tf.reduce_mean(attention_voxel, axis=[2, 3])

        # 6. Return the attention output
        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "max_voxel_grid_size": self.max_voxel_grid_size,
            "ca_steps": self.ca_steps,
            "ca_kernel_size": self.ca_kernel_size,
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
            "gate_locked": self.gate_locked,
        })
        return config


# Block object that applies layernormalization, dropout, and merging of inputs and outputs
@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='VoxelBlock')
class VoxelBlock(tf.keras.layers.Layer):
    """
    A Transformer-style block that wraps the DynamicVoxelAttentionLayer.
    It uses Pre-Layer Normalization and a Gated Residual Connection.
    """

    def __init__(self, d_model, dropout_rate, max_voxel_grid_size=64, ca_steps=5, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.max_voxel_grid_size = max_voxel_grid_size
        self.ca_steps = ca_steps

        # --- Normalization and Dropout ---
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # --- Core Attention Layer ---
        self.attention = DynamicVoxelAttentionLayer(
            d_model=self.d_model,
            max_voxel_grid_size=self.max_voxel_grid_size,
            ca_steps=self.ca_steps,
            name="dynamic_voxel_attention"
        )

        # --- Gated Merge for Residual Connection ---
        self.gated_merge = GatedMergeLayer(d_model)

    def call(self, inputs, training=False):
        # --- Attention Sub-layer with Pre-LN and Gated Stream Merging ---
        residual = inputs
        norm_x = self.layernorm(inputs)

        attn_output = self.attention(norm_x)
        attn_output = self.dropout(attn_output, training=training)

        # Merge the original input and the attention output
        return self.gated_merge([residual, attn_output])

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "dropout_rate": self.dropout_rate,
            "max_voxel_grid_size": self.max_voxel_grid_size,
            "ca_steps": self.ca_steps,
        })
        return config


## Block 4: Linformer:
# Strong at capturing long - range token - to - token relationships, strategically the last layer.
@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='LinformerAttention')
class LinformerAttention(tf.keras.layers.Layer):
    """
    Implements the Linformer attention mechanism for linear complexity.

    This layer projects the Key (K) and Value (V) matrices along the sequence
    dimension to a fixed size `k_proj`. The attention is then computed between
    the original Query (Q) and the projected K/V, resulting in an O(n) complexity
    with respect to sequence length.

    Args:
        d_model (int): The dimension of the input embeddings (e.g., 512, 768).
        k_proj (int): The low-rank dimension to project K and V to. This is the
                      key hyperparameter controlling the efficiency/accuracy trade-off.
                      Must be less than the sequence length.
        kernel_initializer (str, optional): Initializer for the dense layers.
                                            Defaults to 'glorot_uniform'.
    """

    def __init__(self, d_model, k_proj, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.k_proj = k_proj
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

        # Standard Q, K, V projections
        self.q_dense = tf.keras.layers.Dense(d_model, kernel_initializer=self.kernel_initializer)
        self.k_dense = tf.keras.layers.Dense(d_model, kernel_initializer=self.kernel_initializer)
        self.v_dense = tf.keras.layers.Dense(d_model, kernel_initializer=self.kernel_initializer)

        # The core of Linformer: Low-rank projections for K and V.
        # These layers project the SEQUENCE_LENGTH dimension.
        self.k_projection = tf.keras.layers.Dense(k_proj, kernel_initializer=self.kernel_initializer)
        self.v_projection = tf.keras.layers.Dense(k_proj, kernel_initializer=self.kernel_initializer)

        # Final output projection to stabilize training
        self.output_dense = tf.keras.layers.Dense(d_model, kernel_initializer=self.kernel_initializer)

    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length, d_model)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # 1. Generate Q, K, V from the input
        q = self.q_dense(inputs)  # (batch_size, sequence_length, d_model)
        k = self.k_dense(inputs)  # (batch_size, sequence_length, d_model)
        v = self.v_dense(inputs)  # (batch_size, sequence_length, d_model)

        # 2. Project K and V to the low-rank dimension `k_proj`
        # The original paper uses E @ K^T, where E is (k_proj, seq_len).
        # To implement this with a Dense layer, we transpose K and V,
        # apply the Dense layer to the sequence dimension, and transpose back.
        # K shape: (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        k_transposed = tf.transpose(k, perm=[0, 2, 1])
        # k_proj_transposed shape: (batch, d_model, k_proj)
        k_proj_transposed = self.k_projection(k_transposed)
        # k_proj shape: (batch, k_proj, d_model)
        k_proj = tf.transpose(k_proj_transposed, perm=[0, 2, 1])

        # V shape: (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        v_transposed = tf.transpose(v, perm=[0, 2, 1])
        # v_proj_transposed shape: (batch, d_model, k_proj)
        v_proj_transposed = self.v_projection(v_transposed)
        # v_proj shape: (batch, k_proj, d_model)
        v_proj = tf.transpose(v_proj_transposed, perm=[0, 2, 1])

        # 3. Compute Scaled Dot-Product Attention
        # q shape: (batch_size, sequence_length, d_model)
        # k_proj shape: (batch_size, k_proj, d_model)
        # scores shape: (batch_size, sequence_length, k_proj)
        scores = tf.matmul(q, k_proj, transpose_b=True)

        # Scale scores
        scaled_scores = scores / tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Attention weights
        attention_weights = tf.nn.softmax(scaled_scores, axis=-1)

        # 4. Apply attention weights to the projected Value
        # attention_weights shape: (batch_size, sequence_length, k_proj)
        # v_proj shape: (batch_size, k_proj, d_model)
        # context shape: (batch_size, sequence_length, d_model)
        context_vector = tf.matmul(attention_weights, v_proj)

        # 5. Final linear projection
        output = self.output_dense(context_vector)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "k_proj": self.k_proj,
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
        })
        return config


# Block object that applies LayerNorm, Dropout, and a gated skip connection 
# between the input and the attention output:
@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='LinformerBlock')
class LinformerBlock(tf.keras.layers.Layer):
    """
    A Transformer Block using Pre-Layer Normalization and the LinformerAttention layer.
    This version uses a GATING mechanism for stream merging.
    """

    def __init__(self, d_model, k_proj, dff, dropout_rate=0.1, ffn_dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.k_proj = k_proj
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.ffn_dropout_rate = ffn_dropout_rate

        # --- Attention Sub-layer ---
        self.attention = LinformerAttention(
            d_model=d_model,
            k_proj=k_proj,
            name="linformer_attention"
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # --- Stream Merging Layer (GATING) ---
        # *** CHANGE: Use the standard GatedMergeLayer for consistency ***
        self.gate = GatedMergeLayer(d_model)

        # --- Feed-Forward Network (FFN) Sub-layer ---
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch, seq_len, d_model)
        ], name="feed_forward_network")
        self.dropout2 = tf.keras.layers.Dropout(ffn_dropout_rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        # --- Attention Sub-layer with Pre-LN and Gated Stream Merging ---
        # 1. Normalize inputs
        norm_x = self.layernorm1(inputs)
        # 2. Apply attention
        attn_output = self.attention(norm_x)
        # 3. Apply dropout
        attn_output = self.dropout1(attn_output, training=training)

        # 4. *** CHANGE: GATE the original input and the attention output using the standard layer ***
        # This replaces the old manual gating logic.
        merged_output = self.gate([inputs, attn_output])

        # --- Feed-Forward Sub-layer with Pre-LN and Residual ---
        # 1. Normalize the output of the merged stream
        norm_merged = self.layernorm2(merged_output)
        # 2. Apply FFN
        ffn_output = self.ffn(norm_merged)
        # 3. Apply dropout
        ffn_output = self.dropout2(ffn_output, training=training)
        # 4. Residual connection
        final_output = merged_output + ffn_output

        return final_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "k_proj": self.k_proj,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
            "ffn_dropout_rate": self.ffn_dropout_rate,
        })
        return config

# Adapter to project the embedded output back to a linear projection (BAT)
# Add this class to your cerebrosllmutils/llm_utils.py file
@tf.keras.utils.register_keras_serializable(package='cerebrosllmutils', name='AdapterBlock')
class AdapterBlock(tf.keras.layers.Layer):
    """
    A block to reduce the dimensionality of a sequence from (batch, seq_len, d_model)
    to (batch, seq_len) using a learned gating mechanism.

    This block normalizes the input, applies dropout, creates a per-token scalar gate,
    applies the gate, and then sums the features to produce a single scalar per token.
    """
    def __init__(self, d_model, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        # --- Sub-layers ---
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        # A dense layer to create a learned gate for each token
        self.token_gate_dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # 1. Normalize and apply dropout to the input stream
        x = self.layernorm(inputs)
        x = self.dropout(x, training=training)

        # 2. Create a learned scalar gate for each token in the sequence.
        # Shape: (BATCH_SIZE, SEQUENCE_LENGTH, 1)
        token_gates = self.token_gate_dense(x)

        # 3. Apply the gate to the normalized features using tf.multiply
        # Shape: (BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_DIM)
        gated_sequence = tf.multiply(x, token_gates)

        # 4. Reduce the feature dimension for each token to a single scalar.
        # Shape: (BATCH_SIZE, SEQUENCE_LENGTH)
        flattened_output = tf.reduce_sum(gated_sequence, axis=-1)

        return flattened_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "dropout_rate": self.dropout_rate,
        })
        return config
