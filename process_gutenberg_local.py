"""
Gutenberg Dataset Processor with Local GPU Model
Processes Gutenberg samples using llama-cpp for inference.

Usage:
python process_gutenberg_local.py --min_index 0 --max_index 1000
"""
from pathlib import Path
from typing import List, Dict
import json
import gc
import os
import argparse
import time
from llama_cpp import Llama
from transformers import AutoTokenizer
import numpy as np

# Configuration
INPUT_DIR = "gutenberg-raw-samples"
OUTPUT_DIR = "gutenberg-processed-samples-qwen"
MIN_TOKENS_PER_SAMPLE = 77
MAX_TOKENS_PER_SAMPLE = 122
MAX_CTX = 8192  # Context window
MAX_INPUT_TOKENS = 6000  # Maximum tokens for input text (leave room for prompt overhead)

# Local Model Configuration - Qwen3-Coder-30B Q5_K_M
MODEL_REPO = "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
MODEL_FILE = "Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf"
TOKENIZER_CHECKPOINT = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Compatible Qwen tokenizer

# GPU Configuration
USE_GPU = True
GPU_LAYERS = 25  # Reduce layers to fit in memory with 8K context
N_THREADS = int(os.environ.get("LLAMA_CPU_THREADS", os.cpu_count() or 8))

# Global instances
llm = None
tokenizer = None


def initialize_model():
    """Initialize llama-cpp and tokenizer."""
    global llm, tokenizer
    
    print("Initializing Qwen 3 30B with llama-cpp...")
    
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/blobs/4b78837bbec5ee248e4a5642bf608b6793721af41b92589e40c8da0bce58b907")
    
    llm = Llama(
        model_path=model_path,
        n_ctx=MAX_CTX,
        n_gpu_layers=GPU_LAYERS if USE_GPU else 0,
        n_threads=N_THREADS,
        verbose=True,  # Enable verbose to see what's happening
        flash_attn=False,  # Disable flash attention
    )
    gc.collect()
    
    print(f"  ✓ Qwen 3 30B loaded ({GPU_LAYERS} GPU layers)")
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT, trust_remote_code=True)
    print(f"  ✓ Tokenizer loaded (vocab_size: {len(tokenizer)})")


def send_single_turn_instruction(prompt: str) -> str:
    """Send instruction to llama-cpp model."""
    response = llm(
        prompt,
        temperature=0.2,
        max_tokens=1024,  # Further reduced to fit in context
        stop=None,
    )
    return response["choices"][0]["text"]


def count_tokens(text: str) -> int:
    """Count tokens using Gemma's SentencePiece tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def process_text_with_llm(text: str, min_tokens: int, max_tokens: int) -> List[str]:
    """Process text into training samples using local LLM."""
    # Truncate text if too long to fit in context
    text_tokens = count_tokens(text)
    if text_tokens > MAX_INPUT_TOKENS:
        # Truncate to fit within budget
        tokens = tokenizer.encode(text, add_special_tokens=False)
        truncated_tokens = tokens[:MAX_INPUT_TOKENS]
        text = tokenizer.decode(truncated_tokens)
        print(f"    (truncated from {text_tokens} to {MAX_INPUT_TOKENS} tokens)")
    
    prompt = f"""Extract clean training samples from this text. Each sample MUST be {min_tokens}-{max_tokens} tokens (count tokens precisely).

CRITICAL RULES:
1. Token count: MINIMUM {min_tokens}, MAXIMUM {max_tokens} tokens per sample - COUNT CAREFULLY
2. Target average: 50-60 tokens (2-3 sentences typically)
3. Clean English prose - no citations, URLs, footnotes
4. Complete sentences with proper punctuation
5. Natural endings - no mid-sentence cuts
6. Output format: Python list ONLY: ["sample1", "sample2", ...]

EXAMPLES OF CORRECT LENGTH ({min_tokens}-{max_tokens} tokens):
Example: "Serious doubts exist about the fourth commandment's binding nature. Thousands of congregations recite prayers after hearing it read, asking God to incline their hearts to keep this law. This prayer may express desire for grace or amount to a solemn mockery."

TOO SHORT (under {min_tokens} tokens): "The boy was curious."
TOO LONG (over {max_tokens} tokens): Very long paragraphs spanning many sentences

TEXT TO PROCESS:
{text}

OUTPUT (Python list only):
"""
    
    response = send_single_turn_instruction(prompt)
    
    # Parse Python list from response
    start_idx = response.find('[')
    end_idx = response.rfind(']') + 1
    
    if start_idx != -1 and end_idx > start_idx:
        list_str = response[start_idx:end_idx]
        import ast
        try:
            samples = ast.literal_eval(list_str)
            if isinstance(samples, list):
                return [s for s in samples if isinstance(s, str)]
        except (SyntaxError, ValueError):
            pass
    
    return []


def validate_samples(samples: List[str]) -> Dict:
    """Validate samples and return statistics."""
    if not samples:
        return {"valid": False, "error": "No samples provided"}
    
    token_counts = [count_tokens(sample) for sample in samples]
    token_array = np.array(token_counts)
    
    stats = {
        "num_samples": len(samples),
        "max_tokens": int(token_array.max()),
        "min_tokens": int(token_array.min()),
        "mean_tokens": float(token_array.mean()),
        "std_tokens": float(token_array.std()),
        "over_limit": int(np.sum(token_array > MAX_TOKENS_PER_SAMPLE)),
        "valid": True
    }
    
    return stats


def get_processed_samples() -> set:
    """Get set of already processed sample indices."""
    if not os.path.exists(OUTPUT_DIR):
        return set()
    
    processed = set()
    for filename in os.listdir(OUTPUT_DIR):
        if filename.startswith('sample_') and filename.endswith('_processed.json'):
            try:
                sample_num = int(filename.split('_')[1])
                processed.add(sample_num)
            except (IndexError, ValueError):
                continue
    return processed


def get_input_samples() -> List[Path]:
    """Get list of input sample files sorted by index."""
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        return []
    
    samples = sorted(input_path.glob("sample_*.json"))
    return samples


def main(min_index: int = 0, max_index: int = None):
    """Main processing pipeline."""
    print("=" * 70)
    print("Gutenberg Dataset Processor - Qwen 3 30B Edition")
    print("=" * 70)
    
    # Initialize model
    print(f"\n[1/4] Initializing local model")
    initialize_model()
    
    # Get input samples
    print(f"\n[2/4] Scanning input samples from {INPUT_DIR}/")
    input_samples = get_input_samples()
    
    if not input_samples:
        print(f"  ✗ No samples found in {INPUT_DIR}/")
        return
    
    print(f"  ✓ Found {len(input_samples)} raw samples")
    
    # Filter by index range
    if max_index is None:
        max_index = len(input_samples)
    else:
        max_index = min(max_index, len(input_samples))
    
    if min_index >= max_index:
        print(f"  ✗ Invalid range: min_index={min_index}, max_index={max_index}")
        return
    
    input_samples = input_samples[min_index:max_index]
    print(f"  Processing samples {min_index} to {max_index - 1}")
    
    # Check for already processed
    print(f"\n[3/4] Checking for existing processed samples")
    processed_samples = get_processed_samples()
    if processed_samples:
        print(f"  📂 Found {len(processed_samples)} already processed - will skip")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process samples
    print(f"\n[4/4] Processing samples with local GPU")
    print(f"  Token limit: {MAX_TOKENS_PER_SAMPLE}")
    
    start_time = time.time()
    total_samples_generated = 0
    processed_count = 0
    
    for sample_file in input_samples:
        sample_idx = int(sample_file.stem.split('_')[1])
        
        if sample_idx in processed_samples:
            continue
        
        # Load input
        with open(sample_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        sample_start = time.time()
        
        # Process
        samples = process_text_with_llm(input_data['text'], MIN_TOKENS_PER_SAMPLE, MAX_TOKENS_PER_SAMPLE)
        
        sample_time = time.time() - sample_start
        processed_count += 1
        
        if samples:
            stats = validate_samples(samples)
            
            output_file = os.path.join(OUTPUT_DIR, f"sample_{sample_idx:05d}_processed.json")
            output_data = {
                "source_index": sample_idx,
                "samples": samples,
                "statistics": stats,
                "processing_time": sample_time
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            total_samples_generated += len(samples)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_count
            remaining = (max_index - min_index - processed_count) * avg_time
            eta_mins = remaining / 60
            
            print(f"  Sample {sample_idx}: ✓ {len(samples)} samples " +
                  f"(tokens: {stats['min_tokens']}-{stats['max_tokens']}, " +
                  f"mean={stats['mean_tokens']:.1f}) " +
                  f"[{sample_time:.1f}s] " +
                  f"Progress: {processed_count}/{max_index - min_index} " +
                  f"ETA: {eta_mins:.1f}m")
        else:
            print(f"  Sample {sample_idx}: ✗ No samples generated [{sample_time:.1f}s]")
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Samples processed: {processed_count}")
    print(f"Total training samples generated: {total_samples_generated}")
    print(f"Processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    if processed_count > 0:
        print(f"Average time per sample: {total_time/processed_count:.2f}s")
    print(f"\nOutput: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Gutenberg samples with local GPU")
    parser.add_argument("--min_index", type=int, default=0, 
                        help="Minimum sample index to process (default: 0)")
    parser.add_argument("--max_index", type=int, default=None, 
                        help="Maximum sample index to process (default: all samples)")

    args = parser.parse_args()
    main(args.min_index, args.max_index)
