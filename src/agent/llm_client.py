"""Hugging Face Inference API client for the agent's intelligence layer.

We use ``huggingface_hub.InferenceClient`` because:
* it matches the project's existing HF Spaces deployment context
* it works with a free token, no credit card required
* the same client supports many open chat-completions models, so we can
  swap models without changing call sites.

Design notes:
* Token is read from ``HF_TOKEN`` (env or Streamlit secrets). If missing,
  the client enters *degraded mode*: every method returns an "ok=False"
  response and the orchestrator skips LLM-driven tools, falling back to
  the deterministic Phase-1 path. This keeps the app usable offline.
* Every prompt asks for strict JSON output and we parse defensively —
  any JSON parse failure is reported as a low-confidence result rather
  than crashing the agent.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agent.policies import Policies


@dataclass
class LlmResponse:
    ok: bool
    text: str = ""
    parsed: Optional[Any] = None
    error: Optional[str] = None
    model: Optional[str] = None


def _extract_json(text: str) -> Optional[Any]:
    """Best-effort JSON extraction. Models often wrap JSON in markdown
    fences or add commentary; tolerate both."""
    if not text:
        return None
    # Direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip ```json fences.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try to grab the first balanced {...} or [...] block.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


class LlmClient:
    """Thin wrapper over ``huggingface_hub.InferenceClient``."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        token: Optional[str] = None,
        policies: Optional[Policies] = None,
    ) -> None:
        self.policies = policies or Policies()
        self.model = model or os.environ.get("AGENT_LLM_MODEL") or self.policies.llm_default_model
        if token:
            self.token, self._token_source = token, "explicit"
        elif os.environ.get("HF_TOKEN"):
            self.token, self._token_source = os.environ["HF_TOKEN"], "HF_TOKEN"
        elif os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
            self.token, self._token_source = os.environ["HUGGINGFACEHUB_API_TOKEN"], "HUGGINGFACEHUB_API_TOKEN"
        else:
            self.token, self._token_source = None, None
        self._client = None
        self._init_error: Optional[str] = None
        if self.token:
            try:
                from huggingface_hub import InferenceClient  # local import — optional dep
                self._client = InferenceClient(model=self.model, token=self.token, timeout=self.policies.llm_request_timeout_s)
            except ImportError as exc:
                self._init_error = f"huggingface_hub not installed: {exc}"
            except Exception as exc:
                self._init_error = f"InferenceClient init failed: {exc}"

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def status_message(self) -> str:
        if self.available:
            return f"LLM ready: {self.model}"
        if not self.token:
            return "LLM unavailable: HF_TOKEN not set — agent will run in deterministic mode."
        return f"LLM unavailable: {self._init_error or 'unknown error'}"

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LlmResponse:
        """Send a chat-completion request and parse the assistant reply
        as JSON. Failure modes are returned as ``ok=False`` rather than
        raising."""
        if not self.available:
            return LlmResponse(ok=False, error=self.status_message, model=self.model)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            completion = self._client.chat_completion(  # type: ignore[union-attr]
                messages=messages,
                max_tokens=int(max_new_tokens or self.policies.llm_max_new_tokens),
                temperature=float(temperature if temperature is not None else self.policies.llm_temperature),
            )
            choice = completion.choices[0]
            content = (choice.message.content or "").strip()
        except Exception as exc:
            return LlmResponse(ok=False, error=f"{type(exc).__name__}: {exc}", model=self.model)

        parsed = _extract_json(content)
        return LlmResponse(
            ok=parsed is not None,
            text=content,
            parsed=parsed,
            error=None if parsed is not None else "Could not parse JSON from LLM response.",
            model=self.model,
        )
