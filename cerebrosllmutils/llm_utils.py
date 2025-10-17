"""
Utility package with LLM components.



"""



def prepare_data(data: list[str], tokenizer, max_seq_length: int = 1024, prompt_length: int=1):
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
    data : list of str
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
        - all_input_ids (list of list of int): list[list[int]] Token IDs for each input sequence, shaped 
          [num_samples, max_seq_length].
        - all_labels (list of list of int): list[list[int]] One-hot encoded labels for next-token prediction,
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

    pad_token_id = tokenizer.pad_token_id

    # Tokenize all data at once for efficiency
    tokenized_data = tokenizer(
        data,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=False  # We'll handle special tokens manually
    )
    vocab_size = len(tokenizer)

    # Get the token ID for </prompt>
    end_prompt_token_id = tokenizer.encode("</prompt>", add_special_tokens=False)[0]

    # Process each sample
    for sample_tokens in tokenized_data['input_ids']:
        # Find the index of </prompt> token
        try:
            end_prompt_index = sample_tokens.index(end_prompt_token_id)
        except ValueError:
            # If </prompt> not found, treat sample as a non-instruct sample
            end_prompt_index = (
                        PROMPT_LENGTH - 1)  # int(np.ceil(len(sample_tokens) * (1/3)))  # 0 ## 1. Give it a fair starting place to predict the next word 2. reduce the number of expanded samples 

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
