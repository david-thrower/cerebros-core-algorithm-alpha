"""
generators.py — Text generation and prompt-inversion utilities.

Extracted from prototype_work_product_data_engineering_pipeline_mlflow.py.
Provides both LLM-backed and deterministic (tokenizer/heuristic) generation
for synthetic instruct-sample creation.
"""
from __future__ import annotations

import os
import re
from math import ceil
from typing import Any, List


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-0.8B"


def _env_bool(key: str, default: bool = True) -> bool:
    value = os.getenv(key, str(default)).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "`"}:
        return text[1:-1].strip()
    return text


def clean_generated_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^(sure|here(?:'s| is)|example prompt:|prompt:|reasoning:)\s*", "", text, flags=re.I)
    text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = strip_wrapping_quotes(text)
    return normalize_text(text)


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

def build_text_generation_pipeline():
    """Lazy-import the Hugging Face pipeline only when actually needed."""
    from transformers import pipeline  # lazy import

    model_id = os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL)
    max_new_tokens = int(os.getenv("MAX_SEQ_LEN", "500"))
    temperature = float(os.getenv("GEN_TEMPERATURE", "0.7"))
    top_p = float(os.getenv("GEN_TOP_P", "0.95"))
    top_k = int(os.getenv("GEN_TOP_K", "50"))

    pipe = pipeline(
        task="text-generation",
        model=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
    )
    return pipe


def draft_single_text(pipe: Any, prompt: str) -> str:
    """
    Single-turn generation, matching the prototype notebook's structure:
    a single user message sent through a chat-capable text generation pipeline.
    """
    messages = [
        {"role": "user", "content": prompt},
    ]
    raw_response = pipe(text=messages)

    # Try the notebook's expected shape first
    try:
        return clean_generated_text(raw_response[-1]["generated_text"][-1]["content"])
    except Exception:
        pass

    # Fallbacks for alternate pipeline output formats
    if isinstance(raw_response, list) and raw_response:
        item = raw_response[0]
        if isinstance(item, dict):
            if "generated_text" in item and isinstance(item["generated_text"], str):
                return clean_generated_text(item["generated_text"])
            if "generated_text" in item and isinstance(item["generated_text"], list):
                try:
                    return clean_generated_text(item["generated_text"][-1]["content"])
                except Exception:
                    pass

    return clean_generated_text(str(raw_response))


def llm_reverse_engineer_prompt(
    pipe: Any,
    seed_sample: str,
    target_seq_len: int,
) -> str:
    """
    Port of the work-product prompt inversion prompt from the prototype notebook.
    """
    system_prompt = f"""
# Your task is to synthetically **impute** an **example** of a prompt that a user may have written to arrive at the example AI assistant generated response.

## It should be no more than {ceil(target_seq_len * 0.7)} tokens in length and should be as concise as possible. Important context:

1. The example prompt will be part of a synthetic instruct training set to fine tune a small language model just like you into a fully personalized small language model that "already knows" the institutional knowledge user's personal know how.
2. The example prompt should **not** be very detailed. The purpose of this is to train this model to take very concise instructions like "Write an invoice for [customer's name] for [SKU #] for service on [DATE]" and "already know the details" which a user would otherwise have to spend all day explaining to an LLM or setting up an elaborate RAG.
3. This task is part of a data pipeline that takes the examples of questions and answers a user uploaded to train this custom model for a personalized AI assistant. This pipeline up-samples the data to compensate for the small volume of data they have to upload.
4. Only generate one example prompt. Hard coded logic will iterate this call as necessary to control how many samples are generated.
5. You are only generating a plausible user prompt for the provided work product. Please do not also generate a response nor think ... /think reasoning. Separation of concerns is applied in this pipeline, a separate function will do that.

## Follow these best practices:

1. Make no comments like "sure, here is your example prompt ..."
2. Leave no tags like "Topic: ..."
3. Do not interpolate the text with extraneous word counts or token counts. Do NOT include "(Word count: 687)", "estimated tokens: 1234", or anything like it.
4. Use a variety of writing styles ranging from formal adult conversations to casual conversations like 2 teenagers chatting.
5. You want to add nothing to the content that would confuse or increase the amount of data needed to understand this. Do not make up anything that was not asserted in the original, unless you are certain it is factual.
6. Budget tokens to approximately fit this window and start and end on a natural starting point and stopping point.
7. **The example prompt you generate should be a concrete and declarative instruction.**

Input example: "Because you're worth it."
Output example: "Write an example of a spammy company tag line."

## This is the **response from a chat assistant** that your job is to generate **one** plausible user prompt for:

{seed_sample}
"""
    return draft_single_text(pipe, system_prompt)


def llm_reverse_engineer_reasoning(
    pipe: Any,
    seed_response: str,
    target_seq_len: int,
) -> str:
    """
    Port of the reverse-engineer-reasoning prompt from the notebook.
    """
    system_prompt = f"""
# Your task is to synthetically impute a reasoning path (chain-of-thought) that a language model might have used to arrive at the provided response.

## It should be between {ceil(target_seq_len * 0.3)} and {ceil(target_seq_len * 0.7)} tokens in length. Important context:

1. The reasoning path will be part of a synthetic instruct training set to fine tune a small language model into a fully personalized model that shows its work.
2. This task is part of a data pipeline that up-samples user data. When users provide only Q&A pairs without reasoning, we must synthetically generate the reasoning to train the model to think step-by-step.
3. Only generate one reasoning path. Hard coded logic will iterate this call as necessary to control the volume of generated samples.
4. You are only generating the reasoning/thinking steps. Do not generate the prompt/question nor repeat the final response.

## Follow these best practices:

1. Make no comments like "sure, here is the reasoning ..."
2. Leave no tags like "Reasoning: ..." or "Thinking: ..."
3. Do not interpolate the text with extraneous word counts or token counts.
4. Use a variety of reasoning styles: formal logical deduction, casual step-by-step thinking, analytical breakdown, etc.
5. The reasoning should logically connect to the provided response without merely restating it.
6. Budget tokens to fit the window naturally.

## This is the response that your generated reasoning should logically lead to:

{seed_response}
"""
    return draft_single_text(pipe, system_prompt)


# ---------------------------------------------------------------------------
# Deterministic fallback using tokenizer-based keyphrase extraction
# ---------------------------------------------------------------------------

_FALLBACK_TOKENIZER = None

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
    "both", "either", "neither", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "about", "above", "below", "between",
    "up", "down", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "it", "its", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "their", "also", "like",
    "well", "much", "many", "still", "already", "even", "back", "way",
    "new", "now", "old", "get", "got", "make", "made", "take", "put",
    "come", "going", "go", "see", "know", "think", "say", "said",
    "use", "used", "using", "one", "two", "first", "last", "next",
    "please", "thank", "thanks", "dear", "hello", "best", "regards",
    "sincerely", "hi", "hey",
})

_PROMPT_FRAMES = (
    "Explain {topic}",
    "What is {topic}?",
    "{topic}",
    "Write about {topic}",
    "How does {topic} work?",
    "Help me understand {topic}",
    "Summarize {topic}",
    "Tell me about {topic}",
    "Break down {topic}",
    "Give me the details on {topic}",
    "What should I know about {topic}?",
    "Describe {topic}",
)


def _get_fallback_tokenizer():
    """Load tokenizer if available. Returns None if offline/unavailable."""
    global _FALLBACK_TOKENIZER
    if _FALLBACK_TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            model_name = os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL)
            _FALLBACK_TOKENIZER = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        except Exception:
            # Tokenizer not cached and no network — fall back to whitespace
            _FALLBACK_TOKENIZER = "unavailable"
    return None if _FALLBACK_TOKENIZER == "unavailable" else _FALLBACK_TOKENIZER


def _extract_words_whitespace(text: str) -> List[str]:
    """Pure-Python word extraction. No tokenizer needed."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _extract_keyphrases(text: str, max_phrases: int = 10) -> List[str]:
    """Extract salient content words. Uses tokenizer if available, else whitespace."""
    from collections import Counter

    tokenizer = _get_fallback_tokenizer()

    if tokenizer is not None:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        text_words_lower = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
        content_words = []
        for tid in token_ids:
            word = tokenizer.decode([tid]).strip().lower()
            if (
                len(word) > 2
                and word.isalpha()
                and word not in _STOPWORDS
                and word in text_words_lower
            ):
                content_words.append(word)
    else:
        content_words = _extract_words_whitespace(text)

    if not content_words:
        content_words = _extract_words_whitespace(text)

    if not content_words:
        return [text[:60]]

    counts = Counter(content_words)
    total = len(content_words)
    threshold = max(2, total // 5)

    scored = sorted(
        counts.items(),
        key=lambda kv: (kv[1], -len(kv[0])),
    )

    phrases: List[str] = []
    seen: set[str] = set()
    for word, count in scored:
        if word not in seen and count <= threshold:
            phrases.append(word)
            seen.add(word)
        if len(phrases) >= max_phrases:
            break

    if len(phrases) < 3:
        for word, _ in counts.most_common():
            if word not in seen:
                phrases.append(word)
                seen.add(word)
            if len(phrases) >= max_phrases:
                break

    return phrases or [text[:60]]


def heuristic_reverse_engineer_prompt(seed_sample: str, target_seq_len: int) -> str:
    """
    Tokenizer-based prompt inversion. Extracts salient keyphrases from the
    work product and composes a short prompt a user might plausibly type.
    Each call returns a different result due to random keyphrase subset selection.
    """
    import random as _rng

    text = normalize_text(seed_sample)
    keyphrases = _extract_keyphrases(text)

    _rng.shuffle(keyphrases)
    n = min(len(keyphrases), _rng.randint(3, 6))
    topic = " ".join(keyphrases[:n])

    frame = _rng.choice(_PROMPT_FRAMES)
    return frame.format(topic=topic)


def heuristic_reasoning_from_response(seed_response: str, synthetic_prompt: str, target_seq_len: int) -> str:
    """
    Tokenizer-based reasoning trace. Extracts key concepts and token count,
    builds a chain-of-thought grounded in actual content.
    """
    text = normalize_text(seed_response)
    tokenizer = _get_fallback_tokenizer()

    if tokenizer is not None:
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
    else:
        token_count = len(text.split())  # approximate

    keyphrases = _extract_keyphrases(text, max_phrases=5)

    steps = [
        f"The work product is {token_count} tokens long.",
        f"Key concepts: {', '.join(keyphrases)}.",
        f"The user prompt \"{synthetic_prompt[:80]}\" targets these topics.",
        f"The response should cover {', '.join(keyphrases[:3])} "
        f"in roughly {token_count} tokens.",
        "Preserve the original scope and factual content.",
    ]

    body = "\n".join(f"- {s}" for s in steps)
    return f"<think>\n{body}\n</think>"


# ---------------------------------------------------------------------------
# Generator wrappers
# ---------------------------------------------------------------------------

class Generators:
    def __init__(self, use_llm: bool, target_seq_len: int):
        self.use_llm = use_llm
        self.target_seq_len = target_seq_len
        self.pipe = build_text_generation_pipeline() if use_llm else None

    def generate_prompt(self, seed_sample: str) -> str:
        if self.use_llm:
            return llm_reverse_engineer_prompt(self.pipe, seed_sample, self.target_seq_len)
        return heuristic_reverse_engineer_prompt(seed_sample, self.target_seq_len)

    def generate_reasoning(self, seed_response: str, synthetic_prompt: str) -> str:
        if self.use_llm:
            return llm_reverse_engineer_reasoning(self.pipe, seed_response, self.target_seq_len)
        return heuristic_reasoning_from_response(seed_response, synthetic_prompt, self.target_seq_len)
