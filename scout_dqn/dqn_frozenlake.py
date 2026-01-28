# DQN with small MLP for RAGEN FrozenLake using the existing env (no env edits)
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any
import json
import re  # Added for ANSI strip

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from ragen.env.frozen_lake.env import FrozenLakeEnv
from ragen.env.frozen_lake.config import FrozenLakeEnvConfig


class FrozenLakeWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human", "ansi"]}

    def __init__(self, env: FrozenLakeEnv):
        super().__init__()
        self._env = env
        self._size = int(self._env.nrow)
        self._tokens = ['P', '_', 'O', 'G', 'X', '√']
        self._token_to_idx = {t: i for i, t in enumerate(self._tokens)}
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self._size * self._size * len(self._tokens),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)

    def _encode_obs(self, text_obs: str) -> np.ndarray:
        # 1. 去除 ANSI 颜色代码
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text_obs = ansi_escape.sub('', text_obs)

        # 2. 清理边框和分割行
        raw_rows = text_obs.split('\n')
        rows = []
        for r in raw_rows:
            clean_r = r.strip()
            # 跳过边框行 (如 +---+) 或空行
            if not clean_r or set(clean_r).issubset({'+', '-', ' '}):
                continue
            # 去除行首行尾的竖线 (如 | P | ->  P )
            clean_r = clean_r.strip('|')
            rows.append(list(clean_r))

        # 3. 构建 Grid
        h = self._size
        w = self._size
        grid = np.zeros((h, w, len(self._tokens)), dtype=np.float32)
        
        # 安全填充，防止索引越界
        for i in range(min(h, len(rows))):
            for j in range(min(w, len(rows[i]))):
                ch = rows[i][j]
                # 修复: 只有在 token 列表中才置1，遇到未知字符(如墙壁)不默认为 Player(0)
                if ch in self._token_to_idx:
                    idx = self._token_to_idx[ch]
                    grid[i, j, idx] = 1.0
                    
        return grid.reshape(-1).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        text_obs = self._env.reset(seed=seed)
        obs = self._encode_obs(text_obs)
        return obs, {}

    def step(self, action: int):
        mapped = int(action) + 1
        text_obs, reward, done, info = self._env.step(mapped)
        obs = self._encode_obs(text_obs)
        terminated = bool(done)
        truncated = False
        return obs, float(reward), terminated, truncated, info or {}

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False

    # Algorithm
    env_id: str = "FrozenLakeDQN"
    total_timesteps: int = 4000_000
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 200_000
    target_network_frequency: int = 2000
    train_frequency: int = 4
    learning_starts: int = 5000

    # Epsilon-greedy
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.2

    # Model size
    hidden_size: int = 128

    # FrozenLake specific
    grid_size: int = 4
    is_slippery: bool = False

    # Eval
    eval_splits: int = 1
    eval_episodes: int = 4000


def make_env(idx, run_name, seed, grid_size, is_slippery, capture_video=False):
    def thunk():
        config = FrozenLakeEnvConfig(size=grid_size, p=0.9, success_rate=0.8, is_slippery=is_slippery, map_seed=seed + idx, render_mode='text')
        env = FrozenLakeEnv(config)
        env = FrozenLakeWrapper(env)
        max_steps = int(grid_size * grid_size * 4)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.ptr = 0
        self.full = False
        self.obs_buf = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.act_buf = np.zeros((capacity,), dtype=np.int64)
        self.rew_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs: np.ndarray, act: int, rew: float, done: bool, next_obs: np.ndarray):
        i = self.ptr
        self.obs_buf[i] = obs
        self.next_obs_buf[i] = next_obs
        self.act_buf[i] = act
        self.rew_buf[i] = rew
        self.done_buf[i] = 1.0 if done else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def can_sample(self, batch_size: int) -> bool:
        return (self.capacity if self.full else self.ptr) >= batch_size

    def sample(self, batch_size: int):
        size = self.capacity if self.full else self.ptr
        idxs = np.random.randint(0, size, size=batch_size)
        return (
            self.obs_buf[idxs],
            self.act_buf[idxs],
            self.rew_buf[idxs],
            self.done_buf[idxs],
            self.next_obs_buf[idxs],
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        try:
            wandb.define_metric("global_step")
            for prefix in ["train/*", "rollout/*", "eval/*", "losses/*", "charts/*", "perf/*"]:
                wandb.define_metric(prefix, step_metric="global_step")
        except Exception:
            pass

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_env(0, run_name, args.seed, args.grid_size, args.is_slippery, args.capture_video)()
    obs_shape = env.observation_space.shape
    act_dim = env.action_space.n

    policy_net = QNetwork(int(np.prod(obs_shape)), act_dim, args.hidden_size).to(device)
    target_net = QNetwork(int(np.prod(obs_shape)), act_dim, args.hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    criterion = nn.SmoothL1Loss()

    rb = ReplayBuffer(args.buffer_size, obs_shape)

    exploration_steps = max(1, int(args.exploration_fraction * args.total_timesteps))
    epsilon_by_step = lambda t: args.end_e + (args.start_e - args.end_e) * max(0.0, (exploration_steps - t) / exploration_steps)

    def collect_eval_trajectories(agent_model, make_env_fn, n_episodes, step_tag):
        out_dir = Path(f"runs/{run_name}/trajectories/step_{step_tag}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "trajectories.jsonl"
        env_eval = make_env_fn()
        collected = 0
        summary_returns = []
        summary_success = []
        with out_path.open("w") as f:
            while collected < n_episodes:
                state, _ = env_eval.reset(seed=args.seed + 100000 + collected)
                traj_states = [np.asarray(state).tolist()]
                traj_actions = []
                traj_rewards = []
                traj_dones = []
                traj_success = []
                done = False
                step_count = 0
                max_eval_steps = getattr(env_eval, '_max_episode_steps', None) or int(args.grid_size * args.grid_size * 4)
                while not done:
                    with torch.no_grad():
                        q = agent_model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                        action = int(torch.argmax(q, dim=1).item())
                    next_state, reward, terminated, truncated, info = env_eval.step(action)
                    traj_actions.append(int(action))
                    traj_rewards.append(float(reward))
                    step_count += 1
                    d = bool(terminated) or bool(truncated) or (step_count >= max_eval_steps)
                    traj_dones.append(d)
                    traj_success.append(bool((info or {}).get('success', False)))
                    state = next_state
                    traj_states.append(np.asarray(state).tolist())
                    done = d
                ep_ret = float(sum(traj_rewards))
                ep_succ = bool(any(traj_success))
                record = {
                    "states": traj_states,
                    "actions": traj_actions,
                    "rewards": traj_rewards,
                    "dones": traj_dones,
                    "success": traj_success,
                    "episode_return": ep_ret,
                    "episode_success": ep_succ,
                }
                f.write(json.dumps(record) + "\n")
                collected += 1
                summary_returns.append(ep_ret)
                summary_success.append(1.0 if ep_succ else 0.0)
        env_eval.close()
        try:
            metrics = {
                "global_step": int(step_tag),
                "episodes": int(n_episodes),
                "success_rate": float(np.mean(summary_success)) if len(summary_success) else 0.0,
                "avg_return": float(np.mean(summary_returns)) if len(summary_returns) else 0.0,
                "std_return": float(np.std(summary_returns)) if len(summary_returns) else 0.0,
            }
            with (out_dir / "metrics.json").open("w") as mf:
                json.dump(metrics, mf)
        except Exception as e:
            print(f"Warning: failed to write eval metrics: {e}")

    global_step = 0
    start_time = time.time()

    obs, _ = env.reset(seed=args.seed)
    ep_success_window = []
    ep_return = 0.0
    ep_len = 0

    eval_every_steps = max(1, args.total_timesteps // args.eval_splits)

    while global_step < args.total_timesteps:
        epsilon = epsilon_by_step(global_step)
        if np.random.rand() < epsilon or global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                action = int(torch.argmax(q_values, dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        # 注意：这里 done 仅用于控制循环和 logging，不用于 ReplayBuffer 的逻辑判断
        done = bool(terminated) or bool(truncated)

        # 修复：Buffer 中只存储真正的 termination (死亡或到达)，不存 truncation (超时)
        # 这样 Q-learning 在超时时不会错误地认为价值归零
        rb.add(obs.astype(np.float32), int(action), float(reward), bool(terminated), next_obs.astype(np.float32))

        obs = next_obs
        ep_return += float(reward)
        ep_len += 1
        global_step += 1

        if done:
            succ = bool((info or {}).get('success', False))
            ep_success_window.append(1.0 if succ else 0.0)
            if len(ep_success_window) > 100:
                ep_success_window.pop(0)
            if args.track:
                try:
                    import wandb
                    wandb.log({
                        "global_step": int(global_step),
                        "rollout/success": float(1.0 if succ else 0.0),
                        "rollout/success_rate_100": float(np.mean(ep_success_window)) if len(ep_success_window) > 0 else None,
                        # PPO-compatible episodic keys
                        "train/episodic_return": float(ep_return),
                        "train/episodic_length": int(ep_len),
                        "train/success": float(1.0 if succ else 0.0),
                        "train/success_rate_100": float(np.mean(ep_success_window)) if len(ep_success_window) > 0 else None,
                    }, step=global_step)
                except Exception:
                    pass
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0

        if rb.can_sample(args.batch_size) and (global_step % args.train_frequency == 0) and (global_step > args.learning_starts):
            batch_obs, batch_act, batch_rew, batch_done, batch_next_obs = rb.sample(args.batch_size)
            b_obs = torch.tensor(batch_obs, dtype=torch.float32, device=device)
            b_act = torch.tensor(batch_act, dtype=torch.int64, device=device)
            b_rew = torch.tensor(batch_rew, dtype=torch.float32, device=device)
            b_done = torch.tensor(batch_done, dtype=torch.float32, device=device)
            b_next_obs = torch.tensor(batch_next_obs, dtype=torch.float32, device=device)

            with torch.no_grad():
                next_q = target_net(b_next_obs).max(dim=1)[0]
                target_q = b_rew + args.gamma * (1.0 - b_done) * next_q

            current_q = policy_net(b_obs).gather(1, b_act.view(-1, 1)).squeeze(1)
            loss = criterion(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
            optimizer.step()

            if args.track:
                try:
                    import wandb
                    wandb.log({
                        "global_step": int(global_step),
                        "train/loss": float(loss.item()),
                        "charts/epsilon": float(epsilon),
                        "perf/SPS": int(global_step / (time.time() - start_time)),
                        "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }, step=global_step)
                except Exception:
                    pass

        if global_step % args.target_network_frequency == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if global_step % 1000 == 0:
            sps = int(global_step / (time.time() - start_time))
            sr100 = float(np.mean(ep_success_window)) if len(ep_success_window) > 0 else 0.0
            print(f"Step {global_step} | SPS: {sps} | Epsilon: {epsilon:.3f} | SR@100: {sr100:.3f}")

        if global_step == 1 or (global_step % eval_every_steps == 0):
            try:
                def eval_thunk():
                    return make_env(0, run_name, args.seed + 9999, args.grid_size, args.is_slippery, False)()
                collect_eval_trajectories(policy_net, eval_thunk, n_episodes=args.eval_episodes, step_tag=global_step)
                if args.track:
                    try:
                        import wandb
                        mpath = Path(f"runs/{run_name}/trajectories/step_{global_step}/metrics.json")
                        if mpath.exists():
                            with mpath.open("r") as mf:
                                metrics = json.load(mf)
                            wandb.log({
                                "eval/success_rate": metrics.get("success_rate"),
                                "eval/avg_return": metrics.get("avg_return"),
                                "eval/std_return": metrics.get("std_return"),
                                "eval/episodes": metrics.get("episodes"),
                            }, step=global_step)
                    except Exception:
                        pass
                print(f"Collected {args.eval_episodes} eval trajectories at step {global_step}")
            except Exception as e:
                print(f"Warning: eval trajectory collection failed at step {global_step}: {e}")

    env.close()