"""Claude generations for the custom corpus (Anthropic Python SDK).

Credentials are resolved by the SDK (ANTHROPIC_API_KEY, or an `ant auth login` profile).
Current Claude models do not accept temperature / top_p, so decoding is recorded as "default".
Server-side refusal fallbacks are enabled so a declined request is retried on another model.
"""

from __future__ import annotations

DEFAULT_MODEL = "claude-opus-5-5"


class ClaudeGenerator:
    family = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low"):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def generate(self, prompt: str, temperature: float | None = None, max_tokens: int = 4000) -> tuple[str | None, str]:
        """Return (text or None if refused, decoding label). `temperature` is ignored (not supported)."""
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            return None, "default"
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return text, "default"
