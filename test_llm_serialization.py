
import tensorflow as tf
from transformers import AutoTokenizer
from cerebrosllmutils.llm_utils import (
            RotaryEmbedding,
            split_alternate,
            rotate_half,
            apply_rotary_pos_emb,
            InterleavedRoPE,
            Perplexity,
            CerebrosNotGPTConfig,
            CerebrosNotGPT,
            WarmupCosineDecayRestarts)

def test_serialization(tokenizer_path: str, model_path: str):
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("✅ Tokenizer loaded successfully.")

        # Load the full CerebrosNotGPT model
        generator = tf.keras.models.load_model(model_path)
        print("✅ CerebrosNotGPT model loaded successfully.")

        if not isinstance(generator, CerebrosNotGPT):
            raise TypeError("Loaded model is not an instance of CerebrosNotGPT.")

        # Test generation
        prompt = "In the beginning God created the "
        input_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']

        output_tokens = generator.generate(
            token_ids=input_ids,
            do_sample=True,
            max_new_tokens=10,
            temperature=0.65,
            top_k=50,
            top_p=0.96,
            presence_penalty=1.2,
            frequency_penalty=1.2
        )

        output_text = tokenizer.decode(output_tokens)
        print(f"🧠 (serialized) Prompt: {prompt} Generated Text from Serialized Model: '{output_text}'")

        return True

    except Exception as e:
        print(f"❌ Error during serialization test: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python test_serialization.py <tokenizer_path> <model_path>")
        sys.exit(1)

    tokenizer_path = sys.argv[1]
    model_path = sys.argv[2]

    success = test_serialization(tokenizer_path, model_path)
    sys.exit(0 if success else 1)
