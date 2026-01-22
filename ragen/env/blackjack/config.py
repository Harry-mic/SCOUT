from dataclasses import dataclass
import gymnasium as gym
from ragen.env.base import BaseEnvConfig


@dataclass
class BlackjackEnvConfig(BaseEnvConfig):
    render_mode: str = "text"
    action_lookup: dict = None

    def __post_init__(self):
        if self.action_lookup is None:
            # 1: Stick, 2: Hit
            self.action_lookup = {1: "Stick", 2: "Hit"}
        self.invalid_act = 0
        self.invalid_act_score = 0.0
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(2, start=1)
