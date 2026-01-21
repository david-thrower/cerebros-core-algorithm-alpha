# Train Cerebros LLM in Docker

**Stable Container Version: `davidt101/cerebros-llm:41`**

## Quick Start

```bash
# Make a folder:
mkdir docker-llm-runs
cd docker-llm-runs
mkdir artifacts  # Do NOT cd into the artifacts directory
chmod 777 ./artifacts  # Make sure the container can add, edit, delete files.
```

# Environment Variables:

- **MLFLOW_PORT**
  - Effect: Sets the port on which the internal MLflow tracking server runs. To disable MLflow server: set to 0.
  - Type: int: Valid values **7777** or **0** 
  - Default: 7777
- **DATASET_TO_RUN**
  - Effect: The Hugging Face dataset identifier string to load for training.
  - Type: str
  - Default: "david-thrower/tiny-stories-mini-96-seq-len-50000-samples"
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

## Dataset Selection / format for DATASET_TO_RUN

1. Is a huggingface dataset name in the format "username/repo", example: "HuggingFaceTB/smoltalk2"
2. Has a key ['train']['text']
3. The key duck types as a List[str]
4. The samples tokenize consistent with the MAX_SEQUENCE_LENGTH


# Example Usage

Run the docker container as-is (no need to supply a command or inject a script, the entrypoint will take care of it):

```bash
sudo docker run -it \
  --gpus all \
  --cpus="10" \
  --memory="32g" \
  -p 8989:7777 \
  -v $(pwd)/artifacts:/opt/artifacts \
  -e MLFLOW_PORT=7777 \
  -e DATASET_TO_RUN="david-thrower/tiny-stories-mini-96-seq-len-50000-samples" \
  -e PHASE_I_A_SAMPLES_TO_CREATE=200 \
  -e PHASE_I_B_SAMPLES_TO_CREATE=500 \
  -e PHASE_I_B_SAMPLE_EXPANSION_BATCH_SIZE=100 \
  -e PHASE_I_B_VAL_SPLIT=0.15 \
  -e MAX_SEQ_LENGTH=96 \
  davidt101/cerebros-llm:41
```

