"""Short- and long-term memory for the agent.

Short-term memory holds within-run context (image stats, OCR confidence,
detected doc-type, intermediate decisions) and is discarded when the
Streamlit session ends.

Long-term memory is JSON-backed at ``data/user_prefs.json``: cross-session
user preferences (default font, line spacing, alignment bias) plus simple
counters of accepted/rejected agent decisions, which let the agent prefer
choices the user has previously confirmed.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


@dataclass
class UserPrefs:
    """Cross-session user preferences. Serialised to JSON."""

    default_font: str = "Calibri"
    default_font_size_pt: int = 11
    default_line_spacing: float = 1.15
    preferred_alignment_bias: str = "auto"   # auto | left | center | right
    enable_llm_cleanup: bool = False         # OPT-IN: LLM OCR cleanup (slow, ~30-90s)
    autonomy_level: str = "assisted"         # manual | assisted | autonomous
    # Lightweight learning counters, keyed by decision name → {accepted, rejected}
    decision_feedback: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Per-doc-type formatting overrides learned from edits, e.g.
    # {"letter": {"signature_alignment": "right"}}
    doc_type_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ShortTermMemory:
    """In-process scratchpad shared across one ``run`` invocation."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def update(self, mapping: Dict[str, Any]) -> None:
        self._store.update(mapping)

    def snapshot(self) -> Dict[str, Any]:
        return dict(self._store)

    def clear(self) -> None:
        self._store.clear()


class LongTermMemory:
    """File-backed user preferences + learning signal.

    Thread-safe (Streamlit may serve concurrent sessions). Writes are
    atomic via temp-file + replace.
    """

    _lock = threading.Lock()

    def __init__(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = Path(os.environ.get("AGENT_PREFS_PATH", "data/user_prefs.json"))
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prefs = self._load()
        self._defer_depth = 0
        self._dirty = False

    def _load(self) -> UserPrefs:
        if not self.path.exists():
            return UserPrefs()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Corrupt file — start fresh but preserve the broken one for inspection.
            backup = self.path.with_suffix(".corrupt.json")
            try:
                self.path.replace(backup)
            except OSError:
                pass
            return UserPrefs()

        prefs = UserPrefs()
        for k, v in raw.items():
            if hasattr(prefs, k):
                setattr(prefs, k, v)
        return prefs

    @property
    def prefs(self) -> UserPrefs:
        return self._prefs

    @contextmanager
    def deferred_save(self) -> Iterator["LongTermMemory"]:
        """Coalesce several mutations into a single file write.

        ``with ltm.deferred_save(): ltm.record_feedback(...) ltm.record_*(...)``
        will write to disk exactly once when the outermost context exits.
        """
        self._defer_depth += 1
        try:
            yield self
        finally:
            self._defer_depth -= 1
            if self._defer_depth == 0 and self._dirty:
                self._dirty = False
                self._write()

    def update_prefs(self, **changes: Any) -> None:
        for k, v in changes.items():
            if hasattr(self._prefs, k):
                setattr(self._prefs, k, v)
        self._save()

    def record_feedback(self, decision: str, accepted: bool) -> None:
        bucket = self._prefs.decision_feedback.setdefault(
            decision, {"accepted": 0, "rejected": 0}
        )
        bucket["accepted" if accepted else "rejected"] += 1
        self._save()

    def record_doc_type_override(self, doc_type: str, key: str, value: Any) -> None:
        bucket = self._prefs.doc_type_overrides.setdefault(doc_type, {})
        bucket[key] = value
        self._save()

    def acceptance_rate(self, decision: str) -> Optional[float]:
        bucket = self._prefs.decision_feedback.get(decision)
        if not bucket:
            return None
        total = bucket["accepted"] + bucket["rejected"]
        return bucket["accepted"] / total if total > 0 else None

    def recall_for_doc_type(self, doc_type: str) -> Dict[str, Any]:
        return dict(self._prefs.doc_type_overrides.get(doc_type, {}))

    def _save(self) -> None:
        if self._defer_depth > 0:
            self._dirty = True
            return
        self._write()

    def _write(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(asdict(self._prefs), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self.path)

    def reset(self) -> None:
        self._prefs = UserPrefs()
        self._save()
