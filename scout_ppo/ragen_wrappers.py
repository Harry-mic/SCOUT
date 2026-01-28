"""
Gymnasium-compatible wrappers for RAGEN environments to enable traditional RL training.
These wrappers convert text-based observations to numerical representations suitable for MLP networks.
"""
import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple


class BanditWrapper(gym.Wrapper):
    """
    Wrapper for RAGEN Bandit that uses only observable text.
    Converts text observations to a fixed-size one-hot hash vector.
    Does not alter episode semantics and does not inspect env internals.
    """
    def __init__(self, env, feature_dim_per_name: int = 16):
        super().__init__(env)
        # Two name slots (first/second), each hashed to one-hot of size K
        self.k = feature_dim_per_name
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2 * self.k,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def _parse_names(self, text_obs: str):
        """Extract the two arm names from the prompt text purely via regex/string ops."""
        # Heuristic: look for the segment after "named " and split by " and "
        try:
            anchor = "named "
            if anchor in text_obs:
                segment = text_obs.split(anchor, 1)[1]
                # Cut at newline if present
                segment = segment.split("\n", 1)[0]
                # Now split by " and " to get two names; also strip punctuation
                parts = segment.split(" and ")
                if len(parts) >= 2:
                    name_a = parts[0].strip().strip(' .!?,')
                    name_b = parts[1].strip().strip(' .!?,')
                    return name_a, name_b
        except Exception:
            pass
        # Fallback: no names found
        return "", ""

    def _names_to_vector(self, name_a: str, name_b: str) -> np.ndarray:
        vec = np.zeros(2 * self.k, dtype=np.float32)
        idx_a = (hash(name_a) % self.k)
        idx_b = (hash(name_b) % self.k)
        vec[idx_a] = 1.0
        vec[self.k + idx_b] = 1.0
        return vec

    def reset(self, **kwargs):
        seed = kwargs.get('seed', None)
        mode = kwargs.get('mode', None)
        text_obs = self.env.reset(seed=seed, mode=mode)
        name_a, name_b = self._parse_names(text_obs)
        return self._names_to_vector(name_a, name_b), {}

    def step(self, action):
        ragen_action = int(action) + 1
        text_obs, reward, done, info = self.env.step(ragen_action)
        name_a, name_b = self._parse_names(text_obs)
        terminated = bool(done)
        truncated = False
        return self._names_to_vector(name_a, name_b), reward, terminated, truncated, info


class FrozenLakeWrapper(gym.Wrapper):
    """
    Wrapper for RAGEN FrozenLake environment.
    Converts grid-based text observations to numerical state representation.
    """
    def __init__(self, env):
        super().__init__(env)
        # Bootstrap an observation to determine grid size from text only
        bootstrap_text = self.env.reset()
        flat, _ = self._parse_observation_and_meta(bootstrap_text)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(flat.shape[0],), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        # Serve the bootstrapped obs on first reset without calling env.reset again
        self._bootstrap_obs = flat
        self._bootstrap_ready = True
        
    def _parse_observation_and_meta(self, text_obs: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Parse text observation into numerical state (one-hot grid + player pos)."""
        lines = text_obs.strip().split('\n')
        grid = []
        player_pos = None
        rows = len(lines)
        cols = max(len(line) for line in lines) if rows > 0 else 0
        # Parse grid
        for i, line in enumerate(lines):
            row = []
            for j, char in enumerate(line):
                if char == 'P':  # Player
                    row.append(0)
                    player_pos = (i, j)
                elif char == '_':  # Frozen
                    row.append(1)
                elif char == 'O':  # Hole
                    row.append(2)
                elif char == 'G':  # Goal
                    row.append(3)
                elif char == 'X':  # Player in hole
                    row.append(2)
                    player_pos = (i, j)
                elif char == '√':  # Player on goal
                    row.append(3)
                    player_pos = (i, j)
                else:
                    row.append(1)  # Default to frozen
            grid.append(row)
        # Pad ragged rows if needed
        grid = np.array([r + [1] * (cols - len(r)) for r in grid], dtype=np.int32)
        grid_size = (rows, cols)
        # One-hot encode grid over 4 cell types
        # One-hot encode grid
        one_hot_grid = np.zeros((rows, cols, 4), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                cell_type = grid[i, j]
                one_hot_grid[i, j, cell_type] = 1.0
        # Flatten grid
        flat_grid = one_hot_grid.flatten()
        # Add normalized player position
        if player_pos is None:
            player_pos = (0, 0)
        player_pos_norm = np.array([
            0.0 if rows <= 1 else player_pos[0] / max(1, rows - 1),
            0.0 if cols <= 1 else player_pos[1] / max(1, cols - 1),
        ], dtype=np.float32)
        flat = np.concatenate([flat_grid, player_pos_norm])
        return flat, grid_size
    
    def reset(self, **kwargs):
        # Filter out 'options' parameter that gymnasium passes but RAGEN doesn't support
        if self._bootstrap_ready:
            # First call returns the bootstrapped observation to avoid double reset
            self._bootstrap_ready = False
            return self._bootstrap_obs.copy(), {}
        seed = kwargs.get('seed', None)
        mode = kwargs.get('mode', None)
        text_obs = self.env.reset(seed=seed, mode=mode)
        state, _ = self._parse_observation_and_meta(text_obs)
        return state, {}
    
    def step(self, action):
        # Map action from 0,1,2,3 to 1,2,3,4 (RAGEN uses 1-indexed actions)
        ragen_action = action + 1
        text_obs, reward, done, info = self.env.step(ragen_action)
        state, _ = self._parse_observation_and_meta(text_obs)
        
        terminated = done
        truncated = False
        
        return state, reward, terminated, truncated, info


class SokobanWrapper(gym.Wrapper):
    """
    Wrapper for RAGEN Sokoban environment.
    Converts grid-based text observations to numerical state representation.
    Note: Does not inherit from gym.Wrapper due to old gym vs gymnasium compatibility.
    """
    def __init__(self, env):
        super().__init__(env)
        # Bootstrap an observation to determine room size from text only
        bootstrap_text = self.env.reset()
        flat, rows, cols = self._parse_observation_and_meta(bootstrap_text)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(flat.shape[0],), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.metadata = getattr(env, 'metadata', {})
        self._bootstrap_obs = flat
        self._bootstrap_ready = True
        
    def _parse_observation_and_meta(self, text_obs: str) -> Tuple[np.ndarray, int, int]:
        """Parse text observation into numerical state and return dims."""
        lines = text_obs.strip().split('\n')
        grid = []
        rows = len(lines)
        cols = max(len(line) for line in lines) if rows > 0 else 0
        # Mapping from characters to cell types
        char_to_type = {
            '#': 0,  # wall
            '_': 1,  # empty
            'O': 2,  # target
            '√': 3,  # box on target
            'X': 4,  # box
            'P': 5,  # player
            'S': 6,  # player on target
        }
        
        for line in lines:
            row = []
            for char in line:
                row.append(char_to_type.get(char, 1))  # Default to empty
            grid.append(row)
        # Pad ragged rows
        grid = np.array([r + [1] * (cols - len(r)) for r in grid], dtype=np.int32)
        # One-hot encode grid
        one_hot_grid = np.zeros((rows, cols, 7), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                cell_type = grid[i, j]
                one_hot_grid[i, j, cell_type] = 1.0
        return one_hot_grid.flatten(), rows, cols
    
    def reset(self, **kwargs):
        if self._bootstrap_ready:
            self._bootstrap_ready = False
            return self._bootstrap_obs.copy(), {}
        seed = kwargs.get('seed', None)
        mode = kwargs.get('mode', None)
        text_obs = self.env.reset(seed=seed, mode=mode)
        state, _, _ = self._parse_observation_and_meta(text_obs)
        return state, {}
    
    def step(self, action):
        # Map action from 0,1,2,3 to 1,2,3,4 (RAGEN uses 1-indexed actions)
        ragen_action = action + 1
        text_obs, reward, done, info = self.env.step(ragen_action)
        state, _, _ = self._parse_observation_and_meta(text_obs)
        
        terminated = done
        truncated = False
        
        return state, reward, terminated, truncated, info
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def render(self):
        if hasattr(self.env, 'render'):
            return self.env.render()
        return None
