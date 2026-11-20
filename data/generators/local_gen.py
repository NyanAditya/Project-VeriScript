"""Local open-weight generations (e.g. Llama 3 Instruct) through Hugging Face Transformers.

Gated models such as meta-llama/Meta-Llama-3-8B-Instruct need `huggingface-cli login` and
accepting the licence on the Hub. 8B in fp16 needs ~16 GB VRAM; on an 8 GB GPU pass
--load-in-4bit (requires `pip install bitsandbytes`) or pick a smaller instruct model.
"""

from __future__ import annotations


class LocalGenerator:
    def __init__(self, model: str = "meta-llama/Meta-Llama-3-8B-Instruct", load_in_4bit: bool = False, top_p: float = 0.95):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model
        self.top_p = top_p
        self.family = "llama" if "llama" in model.lower() else model.split("/")[-1].split("-")[0].lower()
        kw = {"torch_dtype": torch.float16, "device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 900) -> tuple[str | None, str]:
        ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt")
        ids = ids.to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_tokens, do_sample=True, temperature=temperature, top_p=self.top_p,
                                      pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        return text, f"T={temperature},top_p={self.top_p}"
