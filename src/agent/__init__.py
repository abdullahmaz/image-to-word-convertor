"""Agentic layer for the Image-to-Word converter.

This package wraps the deterministic Phase-1 pipeline (OCR, layout, docx
build) inside a goal-based learning agent that perceives, decides, acts,
and learns. See ``orchestrator.run`` for the entry point.
"""

from src.agent.orchestrator import AgentOrchestrator, RunResult, AutonomyLevel
from src.agent.memory import ShortTermMemory, LongTermMemory, UserPrefs
from src.agent.tools import ToolRegistry, ToolResult, default_registry
from src.agent.policies import Policies
from src.agent.explainability import RunLog, RunLogEntry

__all__ = [
    "AgentOrchestrator",
    "RunResult",
    "AutonomyLevel",
    "ShortTermMemory",
    "LongTermMemory",
    "UserPrefs",
    "ToolRegistry",
    "ToolResult",
    "default_registry",
    "Policies",
    "RunLog",
    "RunLogEntry",
]
