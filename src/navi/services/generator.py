"""
Chat messages → answer text. Two implementations behind one Protocol, like the embedder:

- LocalGenerator — a small instruct model (Qwen2.5-0.5B-Instruct by default), optionally with
  a LoRA (Low-Rank Adaptation) adapter merged in. Greedy decoding: the same question over the
  same sources gives the same answer — needed for a fair evaluation and for debugging.
- FakeGenerator — no model: cites source [1] with its first sentence. Tests and CI.
"""

import re
from typing import Protocol


class Generator(Protocol):
    name: str

    def generate(self, messages: list[dict[str, str]]) -> str: ...


def pick_device(preferred: str = 'auto') -> str:
    import torch

    if preferred != 'auto':
        return preferred
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class LocalGenerator:
    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        device: str = 'auto',
        max_new_tokens: int = 160,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = pick_device(device)
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Half precision on a GPU (CUDA / Apple MPS) halves memory; the CPU needs float32.
        dtype = torch.float32 if self.device == 'cpu' else torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        if adapter_path:
            from peft import PeftModel

            # Merge the adapter into the weights: one matrix multiply per layer at inference,
            # instead of base + A·B every time.
            model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.name = f'{model_name}+lora' if adapter_path else model_name

    def generate(self, messages: list[dict[str, str]]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors='pt', return_dict=True
        ).to(self.device)
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs['input_ids'].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


_FIRST_SENTENCE = re.compile(r'^(.+?[.!?])(\s|$)', re.S)


class FakeGenerator:
    name = 'fake'

    def generate(self, messages: list[dict[str, str]]) -> str:
        user = messages[-1]['content']
        first = user.split('[1] ', 1)
        if len(first) < 2:
            return "I don't have that information in MedVia's content."
        body = first[1].split('\n', 1)[-1].split('\n\n')[0]
        match = _FIRST_SENTENCE.match(body.strip())
        return f'{(match.group(1) if match else body).strip()} [1]'
