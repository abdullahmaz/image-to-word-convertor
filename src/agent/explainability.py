"""Run-level reasoning trace for transparency / explainability.

Every stage of the agent loop appends a ``RunLogEntry`` describing what
was observed, what was decided, and why. The UI surfaces these in a
"Show reasoning" expander; the same log is persisted as JSONL under
``logs/`` for offline audit.

We never log raw image bytes or full document text — only metadata,
truncated samples, and decision rationales. This is consistent with the
data-protection posture stated in the privacy banner.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


_TEXT_SAMPLE_LIMIT = 240  # chars retained for any text snippet in logs


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= _TEXT_SAMPLE_LIMIT else value[:_TEXT_SAMPLE_LIMIT] + "…"
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    # Fallback: stringify non-serialisable objects so the log never crashes.
    return _sanitize(repr(value))


@dataclass
class RunLogEntry:
    stage: str            # observe | interpret | decide | act | reflect | learn | hitl
    name: str             # short event name, e.g. "tool_extract_text"
    rationale: str        # human-readable why
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    elapsed_ms: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["inputs"] = _sanitize(d["inputs"])
        d["outputs"] = _sanitize(d["outputs"])
        return d


class RunLog:
    """In-memory list of entries with optional JSONL persistence."""

    def __init__(self, run_id: Optional[str] = None, log_dir: Optional[Path] = None) -> None:
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.entries: List[RunLogEntry] = []
        self._persisted_count = 0
        if log_dir is None:
            log_dir = Path(os.environ.get("AGENT_LOG_DIR", "logs"))
        self.log_dir = Path(log_dir)

    def add(self, entry: RunLogEntry) -> None:
        self.entries.append(entry)

    def log(
        self,
        stage: str,
        name: str,
        rationale: str,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        elapsed_ms: Optional[float] = None,
    ) -> RunLogEntry:
        entry = RunLogEntry(
            stage=stage,
            name=name,
            rationale=rationale,
            inputs=inputs or {},
            outputs=outputs or {},
            confidence=confidence,
            elapsed_ms=elapsed_ms,
        )
        self.add(entry)
        return entry

    def render_markdown(self) -> str:
        """Human-readable trace for the Streamlit reasoning expander."""
        lines: List[str] = []
        for i, e in enumerate(self.entries, 1):
            head = f"**{i}. [{e.stage}] {e.name}**"
            if e.confidence is not None:
                head += f"  _(confidence: {e.confidence:.2f})_"
            if e.elapsed_ms is not None:
                head += f"  _({e.elapsed_ms:.0f} ms)_"
            lines.append(head)
            if e.rationale:
                lines.append(f"   - _Why_: {e.rationale}")
            if e.inputs:
                lines.append(f"   - _Inputs_: `{json.dumps(_sanitize(e.inputs), ensure_ascii=False)}`")
            if e.outputs:
                lines.append(f"   - _Outputs_: `{json.dumps(_sanitize(e.outputs), ensure_ascii=False)}`")
        return "\n".join(lines) if lines else "_No agent activity recorded yet._"

    def persist(self) -> Optional[Path]:
        """Append-only write to ``logs/run_<id>.jsonl``. Idempotent —
        only entries added since the previous call are written, so
        callers can ``persist()`` multiple times in a run (e.g. once at
        ``reflect`` and again at ``learn``) without duplicating lines."""
        if len(self.entries) <= self._persisted_count:
            return None
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"run_{self.run_id}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                for e in self.entries[self._persisted_count :]:
                    fh.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
            self._persisted_count = len(self.entries)
            return path
        except OSError:
            return None
