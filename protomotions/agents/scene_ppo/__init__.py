"""Checkpoint-compatible scene residuals for PPO motion tracking."""

from .agent import SceneResidualPPO
from .config import (
    SceneResidualCriticConfig,
    SceneResidualPPOActorConfig,
    SceneResidualPPOAgentConfig,
)
from .model import SceneResidualCritic, SceneResidualPPOActor

__all__ = [
    "SceneResidualCritic",
    "SceneResidualCriticConfig",
    "SceneResidualPPO",
    "SceneResidualPPOActor",
    "SceneResidualPPOActorConfig",
    "SceneResidualPPOAgentConfig",
]
