

# Environment Variables: 

These parameters can be set by setting an environment variable in Docker like the example below.

- **MLFLOW_PORT**
  - Effect: Sets the port on which the internal MLflow tracking server runs. To disable MLflow server: set to 0.
  - Type: int: Valid values **7777** or **0** 
  - Default: 7777
- **DATASET_TO_RUN**
  - Effect: The Hugging Face dataset name for the dataset you want to load for training. The supported format 'user/repo'. 
  - Type: str
  - Default: "david-thrower/tiny-stories-mini-96-seq-len-50000-samples"
  - Notes: (Dataset Compatibility) 
- **PHASE_I_A_SAMPLES_TO_CREATE**
  - Effect The number of samples to use for the Neural Architecture Search (Stage I-a). 	
  - Type: int 
  - Default: 300
- **PHASE_I_B_SAMPLES_TO_CREATE**
  - Effect: The number of samples to use for the main training stage (Stage I-b). 	
  - Type: int
  - Default: 200
- **PHASE_I_B_SAMPLE_EXPANSION_BATCH_SIZE**:
  - Effect: Number of samples expanded in memory. (Higher = more RAM pressure and faster training. Lower = less RAM pressure and a CPU bottleneck expanding the samples.)
  - Type: Int
  - Default: 100
- **PHASE_I_B_VAL_SPLIT**
  - Effect: The fraction of the Stage I-b dataset to reserve for validation (e.g., 0.15 for 15%). 	
  - Type: float
  - Default 0.15
- **MAX_SEQ_LENGTH**
  - Effect: The maximum sequence length for tokenization and model input dimensions. 	
  - Type int
  - Default: 96
- **OWNER**
  - Effect: Metadata for file naming and organizing artifacts.
  - Type: str
  - Default: "cerebros"

## Dataset Selection / format for DATASET_TO_RUN

1. Is a Hugging Face dataset name in the format "username/repo", example: "HuggingFaceTB/smoltalk2"
2. Has a key `['train']['text']`
3. The value for said key duck types as a `List[str]`
4. The samples after being tokenized should be consistent with the `MAX_SEQUENCE_LENGTH` for best performance.

In other words:

- The script loads the dataset as `ds = load_dataset(DATASET_TO_RUN)`. (Must be a valid Hugging Face dataset name.)
- It extracts the training text samples to train with as: `ds_text_column = ds['train']['text']` so this key must exist or it will error out.
- It will extract that as a Python list: `x_list = list(ds_text_column)`, so the result of this must be a `list[str]` which is the format the tokenizer supports.
- It is ideal to use a dataset with text samples of lengths spanning a few tokens to the `MAX_SEQ_LENGTH`. Text can be chunked, but if you have text samples with a natural starting point and stopping point within the model's sequence length, the model can make more sense of the data with less data.

# Example use

```

docker run --gpus all -t \
  -p 5000:7777 \
  -v $(pwd)/artifacts:/opt/artifacts \
  -e MLFLOW_PORT=7777 \
  -e DATASET_TO_RUN="david-thrower/tiny-stories-mini-96-seq-len-50000-samples" \
  -e PHASE_I_A_SAMPLES_TO_CREATE=300 \
  -e PHASE_I_B_SAMPLES_TO_CREATE=200 \
  -e PHASE_I_B_SAMPLE_EXPANSION_BATCH_SIZE=100 \
  -e PHASE_I_B_VAL_SPLIT=0.15 \
  -e MAX_SEQ_LENGTH=96 \
  -e OWNER="cerebros" \
  davidt101/cerebros-llm:latest

```

