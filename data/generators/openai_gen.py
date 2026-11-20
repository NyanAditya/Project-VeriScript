"""GPT-4-class generations through the OpenAI Python SDK (`pip install openai`; OPENAI_API_KEY)."""

from __future__ import annotations


class OpenAIGenerator:
    family = "gpt"

    def __init__(self, model: str = "gpt-4o", top_p: float = 0.95):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.top_p = top_p

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1500) -> tuple[str | None, str]:
        r = self.client.chat.completions.create(
            model=self.model, temperature=temperature, top_p=self.top_p, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return (r.choices[0].message.content or "").strip(), f"T={temperature},top_p={self.top_p}"
