"""Local open-source LLM helper (Qwen2.5 Instruct via Hugging Face Transformers)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

from backend.app.core.config import get_settings


@lru_cache
def load_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = get_settings()
    name = settings.llm_model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    # Prefer fp16 on GPU; fall back to float32 on CPU. No paid API.
    dtype = torch.float16 if use_cuda else torch.float32
    # Newer torch/transformers can materialize meta tensors when using the
    # default low-CPU path; calling `.to(device)` then raises:
    # "Cannot copy out of meta tensor; no data!".
    # Load straight onto the target device (or disable meta init on CPU).
    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=dtype,
            device_map={"": 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer, model, device, name


def build_llm_messages(
    question: str, context_blocks: list[dict[str, Any]]
) -> tuple[str, str, list[dict[str, str]], str]:
    """Build the exact system/user contents sent to Qwen."""
    context = "\n\n".join(
        f"[{i+1}] {block.get('agency', 'Source')} — {block.get('title', 'Untitled')}\n"
        f"URL: {block.get('source_url', '')}\n{block.get('text', '')}"
        for i, block in enumerate(context_blocks)
    ) or "No retrieved official passages were available."
    system = (
        "You are ScamShield US, an educational consumer-fraud assistant. "
        "Use only the provided official passages. Cite sources as [1], [2], etc. "
        "If evidence is weak or missing, say so. Do not claim to know whether a specific "
        "agency contacted the user. Never tell the user to click suspicious links or send money. "
        "Reply in plain language with safe next steps."
    )
    user = f"Official passages:\n{context}\n\nConsumer question:\n{question}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return system, user, messages, context


def generate_answer(question: str, context_blocks: list[dict[str, Any]], max_new_tokens: int = 320) -> dict[str, Any]:
    tokenizer, model, device, name = load_llm()
    system, user, messages, context = build_llm_messages(question, context_blocks)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0][encoded["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {
        "model": name,
        "answer": text,
        "llm_input": {
            "model": name,
            "system": system,
            "user": user,
            "messages": messages,
            "chat_template_prompt": prompt,
            "retrieved_passage_count": len(context_blocks),
            "context_preview": context[:4000],
            "generation": {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            },
        },
        "citations": [
            {
                "n": i + 1,
                "title": block.get("title"),
                "agency": block.get("agency"),
                "source_url": block.get("source_url"),
                "retrieved_date": block.get("retrieved_date"),
            }
            for i, block in enumerate(context_blocks)
        ],
    }
