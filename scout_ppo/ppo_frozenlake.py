# PPO with small MLP for RAGEN FrozenLake using the existing env (no env edits)
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any
import json

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from ragen.env.frozen_lake.env import FrozenLakeEnv
from ragen.env.frozen_lake.config import FrozenLakeEnvConfig

# python /mnt/general/wanghy/RAGEN/cleanrl/cleanrl/ppo_frozenlake_nochangeenv.py --total-timesteps 2000000 --num-envs 8 --num-steps 128 --grid-size 4 --is-slippery --track
class FrozenLakeWrapper(gym.Env):
    """
    Adapter to use ragen FrozenLakeEnv with Gymnasium vector API.
    - Converts text observation to one-hot grid vector (6 tokens per cell).
    - Maps agent actions [0..3] to env actions [1..4].
    - Exposes proper observation_space and action_space.
    """
    metadata = {"render_modes": ["rgb_array", "human", "ansi"]}

    def __init__(self, env: FrozenLakeEnv):
        super().__init__()
        self._env = env
        self._size = int(self._env.nrow)  # square grid
        self._tokens = ['P', '_', 'O', 'G', 'X', '√']
        self._token_to_idx = {t: i for i, t in enumerate(self._tokens)}
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self._size * self._size * len(self._tokens),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)

    def _encode_obs(self, text_obs: str) -> np.ndarray:
        # text_obs is multi-line grid with tokens above
        rows = text_obs.split('\n')
        # handle any accidental extra whitespace
        rows = [list(r) for r in rows if len(r) > 0]
        h = len(rows)
        w = len(rows[0]) if h > 0 else self._size
        grid = np.zeros((h, w, len(self._tokens)), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                ch = rows[i][j]
                idx = self._token_to_idx.get(ch, 0)
                grid[i, j, idx] = 1.0
        return grid.reshape(-1)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        text_obs = self._env.reset(seed=seed)
        obs = self._encode_obs(text_obs)
        return obs, {}

    def step(self, action: int):
        # map 0..3 -> 1..4 for the underlying env
        mapped = int(action) + 1
        text_obs, reward, done, info = self._env.step(mapped)
        obs = self._encode_obs(text_obs)
        terminated = bool(done)
        truncated = False
        # propagate success if present
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
    env_id: str = "FrozenLake"
    total_timesteps: int = 200_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # FrozenLake specific
    grid_size: int = 4
    is_slippery: bool = True

    # runtime filled
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    # eval config to mirror reference script
    eval_splits: int = 2
    eval_episodes: int = 4


def make_env(idx, run_name, seed, grid_size, is_slippery, capture_video=False):
    def thunk():
        config = FrozenLakeEnvConfig(size=grid_size, p=0.9, success_rate = 0.8, is_slippery=is_slippery, map_seed=seed + idx, render_mode='text')
        # import pdb;pdb.set_trace()
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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = int(np.array(envs.single_observation_space.shape).prod())
        hidden = 64
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # envs
    envs = gym.vector.SyncVectorEnv([
        make_env(i, run_name, args.seed, args.grid_size, args.is_slippery, args.capture_video)
        for i in range(args.num_envs)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # start
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    episode_returns = []
    episode_steps = []
    episode_successes = []

    # Eval helper identical to reference
    def collect_eval_trajectories(agent_model, make_env_fn, n_episodes, step_tag):
        out_dir = Path(f"runs/{run_name}/trajectories/step_{step_tag}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "trajectories.jsonl"
        env = make_env_fn()
        collected = 0
        summary_returns = []
        summary_success = []
        with out_path.open("w") as f:
            while collected < n_episodes:
                state, _ = env.reset(seed=args.seed + 100000 + collected)
                traj_states = [state.tolist()]
                traj_actions = []
                traj_rewards = []
                traj_dones = []
                traj_success = []
                done = False
                step_count = 0
                max_eval_steps = getattr(env, '_max_episode_steps', None) or int(args.grid_size * args.grid_size * 4)
                while not done:
                    with torch.no_grad():
                        logits = agent_model.actor(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                        action = int(torch.argmax(logits, dim=1).item())
                    next_state, reward, terminated, truncated, info = env.step(action)
                    traj_actions.append(int(action))
                    traj_rewards.append(float(reward))
                    step_count += 1
                    d = bool(terminated) or bool(truncated) or (step_count >= max_eval_steps)
                    traj_dones.append(d)
                    traj_success.append(bool(info.get('success', False)))
                    state = next_state
                    traj_states.append(state.tolist())
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
        env.close()
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

    eval_every_iters = max(1, args.num_iterations // args.eval_splits)
    for iteration in range(1, args.num_iterations + 1):
        # Anneal LR
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            # Episode stats logging as in reference
            try:
                mask = None
                if isinstance(infos, dict):
                    if "_episode" in infos:
                        mask = np.asarray(infos["_episode"]).astype(bool)
                    elif "episode" in infos and isinstance(infos["episode"], dict) and "_l" in infos["episode"]:
                        mask = np.asarray(infos["episode"]["_l"]).astype(bool)
                if mask is not None and np.any(mask):
                    r_arr = np.asarray(infos.get("episode", {}).get("r", np.zeros_like(mask, dtype=float)))
                    l_arr = np.asarray(infos.get("episode", {}).get("l", np.zeros_like(mask, dtype=int)))
                    succ_arr = np.asarray(infos.get("success", np.zeros_like(mask, dtype=bool))).astype(float)
                    for i in np.where(mask)[0]:
                        ep_r = float(r_arr[i])
                        ep_l = int(l_arr[i])
                        ep_succ = float(succ_arr[i])
                        episode_returns.append(ep_r)
                        episode_steps.append(global_step)
                        episode_successes.append(ep_succ)
                    if args.track:
                        try:
                            import wandb
                            log_dict = {
                                "global_step": int(global_step),
                                "rollout/ep_rew_mean": float(np.mean(r_arr[mask])) if np.any(mask) else None,
                                "rollout/ep_len_mean": float(np.mean(l_arr[mask])) if np.any(mask) else None,
                                "rollout/success_rate": float(np.mean(succ_arr[mask])) if np.any(mask) else None,
                            }
                            if np.any(mask):
                                last_idx = np.where(mask)[0][-1]
                                log_dict.update({
                                    "train/episodic_return": float(r_arr[last_idx]),
                                    "train/episodic_length": int(l_arr[last_idx]),
                                    "train/success": float(succ_arr[last_idx]),
                                    "train/success_rate_100": float(np.mean(episode_successes[-100:])) if len(episode_successes) >= 100 else None,
                                })
                            wandb.log(log_dict, step=global_step)
                        except Exception:
                            pass
            except Exception:
                pass

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # update
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        progress = 100 * iteration / args.num_iterations
        print(f"[{progress:5.1f}%] Iter {iteration:4d}/{args.num_iterations} | "
              f"SPS: {sps:5d} | "
              f"Reward: {rewards.mean().item():6.3f} | "
              f"Value: {values.mean().item():6.3f} | "
              f"VLoss: {v_loss.item():.4f} | "
              f"PLoss: {pg_loss.item():.4f} | "
              f"Ent: {entropy_loss.item():.4f}")
        if args.track:
            try:
                import wandb
                wandb.log({
                    "global_step": int(global_step),
                    "train/value_loss": float(v_loss.item()),
                    "train/policy_loss": float(pg_loss.item()),
                    "train/entropy": float(entropy_loss.item()),
                    "train/old_approx_kl": float(old_approx_kl.item()),
                    "train/approx_kl": float(approx_kl.item()),
                    "train/clipfrac": float(np.mean(clipfracs)),
                    "losses/explained_variance": float(explained_var),
                    "charts/avg_reward": float(rewards.mean().item()),
                    "charts/avg_value": float(values.mean().item()),
                    "perf/SPS": int(sps),
                    "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                }, step=global_step)
            except Exception:
                pass

        # periodic evaluation collection
        if iteration==1 or iteration % eval_every_iters == 0:
            try:
                eval_thunk = make_env(0, run_name, args.seed + 9999, args.grid_size, args.is_slippery, False)
                collect_eval_trajectories(agent, eval_thunk, n_episodes=args.eval_episodes, step_tag=global_step)
                if args.track:
                    try:
                        import json as _json
                        from pathlib import Path as _Path
                        mpath = _Path(f"runs/{run_name}/trajectories/step_{global_step}/metrics.json")
                        if mpath.exists():
                            with mpath.open("r") as mf:
                                metrics = _json.load(mf)
                            wandb.log({
                                "eval/success_rate": metrics.get("success_rate"),
                                "eval/avg_return": metrics.get("avg_return"),
                                "eval/std_return": metrics.get("std_return"),
                                "eval/episodes": metrics.get("episodes"),
                            }, step=global_step)
                    except Exception:
                        pass
                print(f"Collected {args.eval_episodes} eval trajectories at global_step {global_step}")
            except Exception as e:
                print(f"Warning: eval trajectory collection failed at step {global_step}: {e}")

    envs.close()
