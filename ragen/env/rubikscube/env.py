import gymnasium as gym
import numpy as np
from typing import Tuple, Any, Dict, List
from ragen.env.base import BaseDiscreteActionEnv
from .config import RubiksCube2x2Config
from ragen.utils import all_seed

class RubiksCube2x2Env(BaseDiscreteActionEnv, gym.Env):
    """
    2x2 Pocket Cube Environment for LLM-based RL.
    State: 24 integers representing the stickers.
    Reward: +1.0 for solved, -0.01 per step to encourage efficiency.
    """
    def __init__(self, config: RubiksCube2x2Config | None = None):
        BaseDiscreteActionEnv.__init__(self)
        self.config = config or RubiksCube2x2Config()
        self.ACTION_LOOKUP = self.config.action_lookup
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(12, start=1)
        self.render_mode = self.config.render_mode
        self.rng = np.random.default_rng()
        
        # 6 faces, 4 stickers per face.
        # Order: U(0), L(1), F(2), R(3), B(4), D(5)
        # Colors: 0:White, 1:Orange, 2:Green, 3:Red, 4:Blue, 5:Yellow
        self.state = np.zeros(24, dtype=int)
        self.solved_state = np.zeros(24, dtype=int)
        self._init_solved_state()
        
        self.current_step = 0
        
    def _init_solved_state(self):
        # Initialize solved state: 4 of color 0, 4 of color 1, ...
        for i in range(6):
            self.solved_state[i*4 : (i+1)*4] = i

    def reset(self, seed=None, mode=None):
        gym.Env.reset(self, seed=seed)
        with all_seed(seed):
            self.rng = np.random.default_rng(seed)
            # 重置为还原状态
            self.state = self.solved_state.copy()
            self.current_step = 0
            
            # 打乱 (Scramble)
            # 随机执行 N 个有效动作
            # 注意：不建议直接随机打乱色块，那样可能导致无解状态
            depth = self.config.scramble_depth
            for _ in range(depth):
                action = self.rng.integers(1, 13) # 1 to 12
                self._apply_action(action)
                
        return self.render(done=False)

    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        assert action in self.ACTION_LOOKUP, f"Invalid action: {action}"
        info = {"action_is_effective": True, "action_is_valid": True, "success": False}
        
        # 执行动作
        self._apply_action(action)
        self.current_step += 1
        
        # import pdb;pdb.set_trace()
        # 检查是否还原
        # is_solved = np.array_equal(self.state, self.solved_state)
        is_solved = True
        for i in range(6):
            # 获取当前面的 4 个色块
            face_stickers = self.state[i*4 : (i+1)*4]
            # 如果这 4 个色块里包含不只 1 种颜色，说明没还原
            if len(set(face_stickers)) > 1:
                is_solved = False
                break
        # import pdb;pdb.set_trace()
        # 检查是否超时
        truncated = self.current_step >= self.config.max_steps
        
        done = is_solved or truncated
        
        if is_solved:
            reward = 1.0
            info["success"] = True
            msg = "Cube Solved!"
        elif truncated:
            # reward = -1.0 # 超时未解出给惩罚
            reward = 0.0
            info["success"] = False
            msg = "Max steps reached."
        else:
            # reward = -0.05 # 每一步给一点惩罚，鼓励最短路径
            reward = 0.0
            msg = ""
            
        next_obs = self.render(done=done, result_msg=msg)
        
        # 按照你给的blackjack示例，这里返回4个值。
        # 如果后续报错需要5个值，请把 truncated 加回去
        return next_obs, float(reward), done, info

    def _rotate_face_clockwise(self, face_idx):
        # Rotates the 4 stickers on a face clockwise
        # Indices: 0 1
        #          3 2  (Z-order usually: 0 1 2 3. Let's assume row-major 0 1 / 2 3)
        # 0 1 -> 2 0
        # 2 3    3 1
        # Mapping: 0->1, 1->3, 3->2, 2->0
        base = face_idx * 4
        s = self.state
        tmp = s[base + 0]
        s[base + 0] = s[base + 2]
        s[base + 2] = s[base + 3]
        s[base + 3] = s[base + 1]
        s[base + 1] = tmp

    def _apply_action(self, action: int):
        # 1-12 mapping to moves
        # 1:U, 2:U', 3:D, 4:D', 5:L, 6:L', 7:R, 8:R', 9:F, 10:F', 11:B, 12:B'
        
        # Helper to simplify inverses: 3 clockwise = 1 counter-clockwise
        if action % 2 == 0: # Even actions are primes (inverses)
            turns = 3
            base_act = action - 1
        else:
            turns = 1
            base_act = action
            
        # Determine which layer to move based on base_act
        # 1:U, 3:D, 5:L, 7:R, 9:F, 11:B
        
        for _ in range(turns):
            if base_act == 1: self._move_U()
            elif base_act == 3: self._move_D()
            elif base_act == 5: self._move_L()
            elif base_act == 7: self._move_R()
            elif base_act == 9: self._move_F()
            elif base_act == 11: self._move_B()

    # --- Primitive Moves (Clockwise) ---
    # Indices: U(0-3), L(4-7), F(8-11), R(12-15), B(16-19), D(20-23)
    # Layout assumptions per face: 0 1 (top row), 2 3 (bottom row)
    
    def _move_U(self):
        self._rotate_face_clockwise(0) # U face
        # U affects top rows of F, R, B, L
        # F(8,9) <- R(12,13) <- B(16,17) <- L(4,5) <- F(8,9)
        s = self.state
        t0, t1 = s[8], s[9]
        s[8], s[9] = s[12], s[13]
        s[12], s[13] = s[16], s[17]
        s[16], s[17] = s[4], s[5]
        s[4], s[5] = t0, t1

    def _move_D(self):
        self._rotate_face_clockwise(5) # D face
        # D affects bottom rows of F, R, B, L
        # F(10,11) -> L(6,7) -> B(18,19) -> R(14,15) -> F(10,11) (Clockwise looking from bottom)
        # Wait, standard D move is moving the bottom layer RIGHT.
        # F -> R -> B -> L -> F
        s = self.state
        t0, t1 = s[10], s[11]
        s[10], s[11] = s[6], s[7]
        s[6], s[7] = s[18], s[19]
        s[18], s[19] = s[14], s[15]
        s[14], s[15] = t0, t1

    def _move_L(self):
        self._rotate_face_clockwise(1) # L face
        # L affects left cols of U, F, D, B
        # U(0,2) -> F(8,10) -> D(20,22) -> B(19,17) (B is reversed vertical) -> U(0,2)
        s = self.state
        t0, t2 = s[0], s[2]
        s[0], s[2] = s[19], s[17]
        s[19], s[17] = s[20], s[22]
        s[20], s[22] = s[8], s[10]
        s[8], s[10] = t0, t2

    def _move_R(self):
        self._rotate_face_clockwise(3) # R face
        # R affects right cols of U, F, D, B
        # U(1,3) -> B(18,16) -> D(21,23) -> F(9,11) -> U(1,3)
        # Note: B side logic depends on unfolding. Assuming standard:
        # U(1,3) goes to F(9,11)? No, R move pulls F up to U.
        # U -> B -> D -> F -> U
        s = self.state
        t1, t3 = s[1], s[3]
        s[1], s[3] = s[9], s[11]
        s[9], s[11] = s[21], s[23]
        s[21], s[23] = s[18], s[16]
        s[18], s[16] = t1, t3

    def _move_F(self):
        self._rotate_face_clockwise(2) # F face
        # F affects bottom U, left R, top D, right L
        # U(2,3) -> L(7,5) -> D(21,20) -> R(12,14) -> U(2,3)
        # This rotation logic is tricky. Clockwise F:
        # U(2,3) goes to R(12,14) (Top-Left of R becomes Top-Right of R? No)
        # Let's trace visual: Top moves Right.
        # U(2,3) -> R(12,14) (Left side of R)
        # R(12,14) -> D(21,20) (Top side of D, reversed)
        # D(21,20) -> L(7,5) (Right side of L)
        # L(7,5) -> U(2,3)
        s = self.state
        t2, t3 = s[2], s[3]
        s[2], s[3] = s[7], s[5]
        s[7], s[5] = s[21], s[20]
        s[21], s[20] = s[12], s[14]
        s[12], s[14] = t2, t3

    def _move_B(self):
        self._rotate_face_clockwise(4) # B face
        # B affects top U, right L, bottom D, left R
        # U(0,1) -> L(4,6) -> D(23,22) -> R(15,13) -> U(0,1)
        # Clockwise B (looking from back):
        # U(0,1) goes Left (to L).
        # U(0,1) -> L(4,6)
        # L(4,6) -> D(23,22)
        # D(23,22) -> R(15,13)
        # R(15,13) -> U(0,1)
        s = self.state
        t0, t1 = s[0], s[1]
        s[0], s[1] = s[13], s[15]
        s[13], s[15] = s[22], s[23]
        s[22], s[23] = s[6], s[4]
        s[6], s[4] = t0, t1

    def render(self, mode: str | None = None, done: bool = False, result_msg: str = "") -> str:
        # Color Mapping
        colors = {0: 'W', 1: 'O', 2: 'G', 3: 'R', 4: 'B', 5: 'Y'}
        
        def get_face_str(face_idx):
            base = face_idx * 4
            c = [colors[self.state[base+i]] for i in range(4)]
            return f"[{c[0]}, {c[1]}]\n       [{c[2]}, {c[3]}]"

        lines = []
        lines.append("=== Rubik's Cube 2x2 State ===")
        lines.append(f"Step: {self.current_step}/{self.config.max_steps}")
        lines.append("")
        
        # ASCII Net Layout
        #       Up
        # Left Front Right Back
        #      Down
        
        # Flattening slightly for LLM readability (List format is often better than ASCII art for tokens)
        lines.append(f"Up (U):    {get_face_str(0).strip().replace('       ', ' ')}")
        lines.append(f"Left (L):  {get_face_str(1).strip().replace('       ', ' ')}")
        lines.append(f"Front (F): {get_face_str(2).strip().replace('       ', ' ')}")
        lines.append(f"Right (R): {get_face_str(3).strip().replace('       ', ' ')}")
        lines.append(f"Back (B):  {get_face_str(4).strip().replace('       ', ' ')}")
        lines.append(f"Down (D):  {get_face_str(5).strip().replace('       ', ' ')}")
        
        if not done:
            lines.append("")
            lines.append("Available Actions:")
            # 列出部分动作作为提示，或者全部列出
            lines.append("U, U', D, D', L, L', R, R', F, F', B, B'")
            # lines.append("Format: Action <id> (e.g., Action 1 for U, Action 2 for U')")
            lines.append("\nWhat is your next move?")
        else:
            lines.append("")
            lines.append(f"=== Game Over: {result_msg} ===")

        return "\n".join(lines)

    def close(self):
        pass

    def get_all_actions(self):
        return list(self.ACTION_LOOKUP.keys())

if __name__ == "__main__":
    # 实例化配置：这里设置打乱深度为 3，方便你手动还原测试
    config = RubiksCube2x2Config(scramble_depth=1, max_steps=50)
    env = RubiksCube2x2Env(config)
    
    # 打印动作映射表，方便你知道输入 1-12 分别代表什么
    print(f"Action Lookup: {env.ACTION_LOOKUP}")
    
    # 重置并打印初始状态
    obs = env.reset()
    print(obs)
    
    while True:
        # 提示输入
        keyboard = input("\nEnter action (1-12) or 'q' to quit: ")
        if keyboard == 'q':
            break
        
        try:
            action = int(keyboard)
        except ValueError:
            print("Please enter a valid integer.")
            continue

        if action not in env.ACTION_LOOKUP:
            print(f"Invalid action: {action}. Please input 1-12.")
            continue
            
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        # 打印状态和奖励信息
        print(obs) # 这里打印的就是 render 返回的文本 Prompt
        print(f"Reward: {reward}, Done: {done}, Info: {info}")
        
        # 如果游戏结束（还原或超时），自动重置
        if done:
            print("\n=== Episode Ended. Resetting... ===")
            obs = env.reset()
            print(obs)