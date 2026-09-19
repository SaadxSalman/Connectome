"""Provider-agnostic LLM endpoint with graceful degradation.

Engine order for ``LLM_PROVIDER=auto``:

    Groq (if GROQ_API_KEY set)  →  Ollama (if reachable)  →  None

When ``complete`` returns ``None`` every caller falls back to the offline
Reflex-Arc synthesiser, so the nervous system *never* goes dark.
"""

from __future__ import annotations

import json
import time

import httpx

from ..config import Settings


class LLMProvider:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        self._ollama_down_until = 0.0
        self.last_error = ""

    # ── identity ───────────────────────────────────────────────────────────
    @property
    def provider_name(self) -> str:
        p = self.s.llm_provider
        if p == "reflex":
            return "reflex"
        if p == "groq":
            return "groq" if self.s.groq_api_key else "reflex"
        if p == "ollama":
            return "ollama"
        # auto
        if self.s.groq_api_key:
            return "groq"
        return "ollama"

    @property
    def model_name(self) -> str:
        p = self.provider_name
        if p == "groq":
            return self.s.groq_model
        if p == "ollama":
            return self.s.ollama_llm_model
        return "reflex-arc-extractive"

    def describe(self) -> dict:
        return {"provider": self.provider_name, "model": self.model_name,
                "fallback": "reflex-arc-extractive"}

    def _engines(self) -> list[str]:
        p = self.s.llm_provider
        order: list[str] = []
        if p in ("auto", "groq") and self.s.groq_api_key:
            order.append("groq")
        if p in ("auto", "ollama"):
            order.append("ollama")
        return order

    # ── completion ─────────────────────────────────────────────────────────
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str | None:
        """Try each engine in order; return None so callers fall back to reflex."""
        for engine in self._engines():
            try:
                if engine == "groq":
                    return await self._groq(prompt, system, json_mode, max_tokens, temperature)
                if engine == "ollama":
                    if time.time() < self._ollama_down_until:
                        continue  # cooldown after a recent failure
                    return await self._ollama(prompt, system, temperature)
            except Exception as exc:
                self.last_error = f"{engine}: {exc}"
                if engine == "ollama":
                    self._ollama_down_until = time.time() + 30
                continue
        return None

    async def _groq(self, prompt, system, json_mode, max_tokens, temperature) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict = {
            "model": self.s.groq_model,
            "messages": messages,
            "temperature": self.s.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.s.llm_max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = await self._client.post(
            f"{self.s.groq_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.s.groq_api_key}"},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    async def _ollama(self, prompt, system, temperature) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = await self._client.post(
            f"{self.s.ollama_base_url.rstrip('/')}/api/chat",
            json={
                "model": self.s.ollama_llm_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.s.llm_temperature if temperature is None else temperature,
                    "num_predict": self.s.llm_max_tokens,
                },
            },
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    async def aclose(self) -> None:
        await self._client.aclose()


def parse_json_loose(text: str | None) -> dict | None:
    """Extract the first balanced JSON object from an LLM reply (tolerant of
    prose wrappers, code fences and trailing chatter)."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except Exception:
                            break
        start = text.find("{", start + 1)
    return None
