import gymnasium as gym
import numpy as np
from typing import Tuple, Any, Dict, List
from ragen.env.base import BaseDiscreteActionEnv
from .config import BlackjackEnvConfig
from ragen.utils import all_seed

def draw_card(rng: np.random.Generator) -> int:
    # 1 is Ace, 2-9 as is, 10 represents 10/J/Q/K
    card_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 4]
    probs = np.array(weights, dtype=np.float64)
    probs /= probs.sum()
    return int(rng.choice(card_vals, p=probs))

def usable_ace(hand: List[int]) -> bool:
    return 1 in hand and sum(hand) + 10 <= 21

def hand_sum(hand: List[int]) -> int:
    total = sum(hand)
    if 1 in hand and total + 10 <= 21:
        return total + 10
    return total

class BlackjackEnv(BaseDiscreteActionEnv, gym.Env):
    def __init__(self, config: BlackjackEnvConfig | None = None):
        BaseDiscreteActionEnv.__init__(self)
        self.config = config or BlackjackEnvConfig()
        self.ACTION_LOOKUP = self.config.action_lookup
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(2, start=1)
        self.render_mode = self.config.render_mode
        self.rng = np.random.default_rng()
        self.player = []
        self.dealer = []
    
    def reset(self, seed=None, mode=None):
        gym.Env.reset(self, seed=seed)
        with all_seed(seed):
            self.rng = np.random.default_rng(seed)
            self.player = [draw_card(self.rng), draw_card(self.rng)]
            self.dealer = [draw_card(self.rng), draw_card(self.rng)]
        
        # 即使起手21点，Gym的reset也不能返回done，只能通过Observation提示
        return self.render(done=False)

    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        assert action in self.ACTION_LOOKUP, f"Invalid action: {action}"
        info = {"action_is_effective": True, "action_is_valid": True, "success": False}
        
        # 1: Stick, 2: Hit
        if action == 2:  # Hit
            self.player.append(draw_card(self.rng))
            if hand_sum(self.player) > 21:
                # 爆牌 (Bust)
                done = True
                reward = -1.0
                info["success"] = False
                # 爆牌时，render需要知道游戏结束，不再请求动作
                next_obs = self.render(done=True, result_msg="You Busted! (Sum > 21)")
                return next_obs, reward, done, info
            
            # 继续游戏
            next_obs = self.render(done=False)
            return next_obs, 0.0, False, info
            
        else:  # Stick
            # Dealer policy: hit to 17 or more
            while hand_sum(self.dealer) < 17:
                self.dealer.append(draw_card(self.rng))
            
            reward = self._settle()
            done = True
            info["success"] = reward > 0
            
            # 生成结算信息
            if reward > 0: res = "You Won!"
            elif reward < 0: res = "You Lost."
            else: res = "Draw."
            
            next_obs = self.render(done=True, result_msg=res)
            return next_obs, float(reward), done, info

    def _settle(self) -> float:
        p = hand_sum(self.player)
        d = hand_sum(self.dealer)
        if p > 21: return -1.0
        if d > 21: return 1.0
        if p > d: return 1.0
        if p < d: return -1.0
        return 0.0

    def render(self, mode: str | None = None, done: bool = False, result_msg: str = "") -> str:
        p_sum = hand_sum(self.player)
        ua = usable_ace(self.player)
        
        lines = []
        lines.append("=== Blackjack Game State ===")
        lines.append(f"Your Hand: {self.player} (Total: {p_sum}).")
        if ua:
            lines.append("Note: You possess a usable Ace.")
        
        if not done:
            # 游戏进行中，只显示庄家的一张牌
            lines.append(f"Dealer's Visible Card: {self.dealer[0]}.")
            lines.append("")
            lines.append("Available Actions:")
            lines.append("- Action 1: Stick (Stop)")
            lines.append("- Action 2: Hit (Add card)")
            
            # 增加策略提示，防止起手21点还Hit
            if p_sum == 21:
                lines.append("\nWait! You have 21. It is strongly recommended to Stick (Action 1).")
            
            lines.append("\nWhat is your next move?")
        else:
            # 游戏结束，显示全貌
            d_sum = hand_sum(self.dealer)
            lines.append(f"Dealer's Hand: {self.dealer} (Total: {d_sum}).")
            lines.append("")
            lines.append(f"=== Game Over: {result_msg} ===")
            # 此时不再询问 Next move
            
        return "\n".join(lines)

    def close(self):
        pass

    def get_all_actions(self):
        return list(self.ACTION_LOOKUP.keys())