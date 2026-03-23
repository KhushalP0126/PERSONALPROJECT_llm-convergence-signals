import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PHI2_MODEL = "microsoft/phi-2"


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(device: str) -> torch.dtype:
    if device in {"mps", "cuda"}:
        return torch.float16
    return torch.float32


def should_trust_remote_code(model_name: str, allow_remote_code: bool) -> bool:
    if allow_remote_code:
        return True
    return model_name.lower() == PHI2_MODEL


def validate_device(device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but PyTorch cannot access it in this session.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")


def load_model(
    model_name: str,
    device: str,
    allow_remote_code: bool,
):
    trust_remote_code = should_trust_remote_code(model_name, allow_remote_code)
    dtype = resolve_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    model.to(device)
    model.eval()

    if getattr(model.generation_config, "max_length", None) is not None:
        model.generation_config.max_length = None

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def tokenize_prompt(prompt: str, tokenizer, device: str):
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    return {key: value.to(device) for key, value in inputs.items()}


def generate_text(
    prompt: str,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
    temperature: float,
    return_sequence: bool = False,
):
    inputs = tokenize_prompt(prompt=prompt, tokenizer=tokenizer, device=device)

    do_sample = temperature > 0
    generation_kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        sequence_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = sequence_ids[0][prompt_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if return_sequence:
        return text, sequence_ids, prompt_length
    return text
