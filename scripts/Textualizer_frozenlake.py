#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


ACTION_LOOKUP = {1: "Left", 2: "Down", 3: "Right", 4: "Up"}


def infer_grid_dims(state_vec: List[float]) -> Tuple[int, int]:
    """Infer (rows, cols) from flattened one-hot grid length.
    Our PPO wrapper encodes each cell as one-hot over 6 tokens: ['P','_','O','G','X','√'].
    """
    n = len(state_vec)
    assert n % 6 == 0, f"State length {n} not divisible by 6 (channels)"
    n_cells = n // 6
    r = int(math.isqrt(n_cells))
    assert r * r == n_cells, f"Grid is not square: {n_cells} cells"
    return r, r


def decode_state_to_grid_text(state_vec: List[float]) -> str:
    """Decode numeric state vector back to textual grid.

    Encoding per PPO wrapper:
      One-hot per cell over tokens = ['P', '_', 'O', 'G', 'X', '√'] in this order.
      The wrapper already encodes P/X/√ directly in the grid; no separate coords needed.
    """
    tokens = ['P', '_', 'O', 'G', 'X', '√']
    rows, cols = infer_grid_dims(state_vec)
    lines = []
    for i in range(rows):
        row_chars = []
        for j in range(cols):
            base = (i * cols + j) * 6
            cell = state_vec[base: base + 6]
            idx = max(range(6), key=lambda k: cell[k])
            ch = tokens[idx] if 0 <= idx < len(tokens) else '_'
            row_chars.append(ch)
        lines.append("".join(row_chars))
    return "\n".join(lines)


def parse_positions_from_state(state_vec: List[float]):
    """Extract board size, player, goal, and holes positions from one-hot state.
    - Player is where token is one of ['P','X','√'].
    - Goal is where token is 'G'.
    - Holes include all 'O' cells; if player is on hole ('X'), include that cell as a hole as well.
    Returns: (rows, cols, (pr, pc), (gr, gc) or None, holes: List[(r,c)])
    """
    tokens = ['P', '_', 'O', 'G', 'X', '√']
    rows, cols = infer_grid_dims(state_vec)
    player = None
    goal = None
    holes: List[Tuple[int, int]] = []
    for i in range(rows):
        for j in range(cols):
            base = (i * cols + j) * 6
            cell = state_vec[base: base + 6]
            idx = max(range(6), key=lambda k: cell[k])
            if idx == 0 or idx == 4 or idx == 5:  # P or X or √
                player = (i, j)
                if idx == 4:  # X means on a hole
                    holes.append((i, j))
            elif idx == 2:  # O
                holes.append((i, j))
            elif idx == 3:  # G
                goal = (i, j)
    return rows, cols, player, goal, holes


def load_env_instruction_and_cfg(repo_root: Path) -> Tuple[str, int, str, bool]:
    """Load FrozenLake env_instruction, max_tokens, action_sep, enable_think from config.
    Fallbacks are provided if YAML is unavailable.
    """
    default_instruction = (
        "You are solving the FrozenLake puzzle. The observation includes both a symbol grid and zero-indexed coordinates for the start, goal, player, and any holes.\n"
        "Coordinates range from the top-left corner (0, 0) to the bottom-right corner (5, 5).\n"
        "Beware that the ice is slippery, so the agent might slide and end up in an unintended tile.\n"
        "Respond with a sequence of actions such as <answer>Left || Up || Up</answer>.\n"
        "\nThe meaning of each symbol in the state is:\n"
        "P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal\n"
        "Your available actions are:\n"
        "Left, Down, Right, Up\n"
        "You can make up to 25 actions, separated by the action separator \" || \"\n"
    )
    instruction = default_instruction
    max_tokens = 100
    action_sep = "||"
    enable_think = True

    if yaml is None:
        return instruction, max_tokens, action_sep, enable_think

    # envs.yaml
    envs_yaml = repo_root / "config" / "envs.yaml"
    if envs_yaml.exists():
        try:
            with open(envs_yaml, "r", encoding="utf-8") as f:
                envs = yaml.safe_load(f)
            if isinstance(envs, dict) and "FrozenLake" in envs:
                fl = envs["FrozenLake"]
                instruction = fl.get("env_instruction", instruction)
                max_tokens = int(fl.get("max_tokens", max_tokens))
        except Exception:
            pass

    # base.yaml
    base_yaml = repo_root / "config" / "base.yaml"
    if base_yaml.exists():
        try:
            with open(base_yaml, "r", encoding="utf-8") as f:
                base_cfg = yaml.safe_load(f)
            ap = base_cfg.get("agent_proxy", {}) if isinstance(base_cfg, dict) else {}
            action_sep = ap.get("action_sep", action_sep)
            enable_think = bool(ap.get("enable_think", enable_think))
        except Exception:
            pass

    return instruction, max_tokens, action_sep, enable_think


def build_messages_for_episode(
    states: List[List[float]],
    actions: List[int],
    rewards: List[float],
    instruction: str,
    max_tokens: int,
    action_sep: str,
    enable_think: bool,
    max_actions: int,
) -> List[dict]:
    """Construct chat messages mirroring ContextManager format.

    - First system message.
    - One user message containing the instruction and per-turn state blocks.
    - Assistant messages per executed action with tag-only outputs.
    - User messages for rewards.
    """
    messages = [
        {"role": "system", "content": "You're a helpful assistant. "},
        {"role": "user", "content": instruction},
    ]

    total_actions = len(actions)
    # Determine start position from the first state's player
    start_rows, start_cols, start_player, start_goal, start_holes = parse_positions_from_state(states[0]) if states else (0, 0, None, None, [])
    # Append state blocks into the initial user content
    for t, state in enumerate(states):
        grid_text = decode_state_to_grid_text(state)
        rows, cols, player_pos, goal_pos, holes_pos = parse_positions_from_state(state)
        # Start counter from max_actions (e.g., 25) regardless of episode length
        actions_left = max(0, max_actions - t)
        format_prompt = (
            "<think> [Your thoughts] </think> <answer> [your answer] </answer>"
            if enable_think
            else "<answer> [your answer] </answer>"
        )
        length_prompt = f"Max response length: {max_tokens} words (tokens)."

        messages[-1]["content"] += (
            f"\nTurn {t + 1}:\n"
            f"State:\n"
            f"Coordinates:\n"
            f"Board size: {rows} rows x {cols} cols (zero-indexed).\n"
            f"Start: {start_player if start_player is not None else (-1, -1)}\n"
            f"Goal: {goal_pos if goal_pos is not None else (-1, -1)}\n"
            f"Player: {player_pos if player_pos is not None else (-1, -1)}\n"
            f"Holes: {holes_pos}\n"
            f"Grid Map:\n{grid_text}\n"
            f"You have {actions_left} actions left. Always output: {format_prompt}"
            f"with no extra text. Strictly follow this format. {length_prompt}"
        )
        # If action exists for this turn, add assistant + reward
        if t < total_actions:
            # Map RL action (0..3) -> RAGEN action (1..4) -> text
            action_id = actions[t] + 1
            action_name = ACTION_LOOKUP.get(action_id, "unknown")
            if enable_think:
                assistant_text = f"<think></think><answer>{action_name}</answer>"
            else:
                assistant_text = f"<answer>{action_name}</answer>"
            messages.append({"role": "assistant", "content": assistant_text})
            # Reward message
            reward_val = rewards[t] if t < len(rewards) else 0.0
            messages.append({"role": "user", "content": f"Reward:\n{reward_val}\n"})
    # import pdb;pdb.set_trace()
    
    return messages[:-1]


def convert_file(step_dir: Path, output_dir: Path, repo_root: Path, include_failed: bool = False, max_actions: int = 25) -> Path:
    traj_path = step_dir / "trajectories.jsonl"
    metrics_path = step_dir / "metrics.json"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing trajectories.jsonl at {traj_path}")

    instruction, max_tokens, action_sep, enable_think = load_env_instruction_and_cfg(repo_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{step_dir.name}_sft.jsonl"

    # Read global step from metrics if available
    global_step = None
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            global_step = m.get("global_step")
        except Exception:
            pass

    written = 0
    with open(traj_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)
            # Filter if requested
            ep_success = bool(traj.get("episode_success", False))
            if (not include_failed) and (not ep_success):
                continue

            states = traj.get("states", [])
            actions = traj.get("actions", [])
            rewards = traj.get("rewards", [])
            # Filter: keep only episodes with total actions <= max_actions
            if len(actions) > max_actions:
                continue
            messages = build_messages_for_episode(
                states=states,
                actions=actions,
                rewards=rewards,
                instruction=instruction,
                max_tokens=max_tokens,
                action_sep=action_sep,
                enable_think=enable_think,
                max_actions=max_actions,
            )

            record = {
                "messages": messages,
                "meta": {
                    "episode_return": traj.get("episode_return", None),
                    "episode_success": ep_success,
                    "global_step": global_step,
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    if written == 0:
        # Still write an empty file to signal conversion executed
        with open(out_path, "w", encoding="utf-8") as f:
            pass
    return out_path


def find_latest_step_dir(traj_root: Path) -> Path:
    step_dirs = [p for p in traj_root.iterdir() if p.is_dir() and p.name.startswith("step_")]
    if not step_dirs:
        raise FileNotFoundError(f"No step_* directories under {traj_root}")
    # Sort by numeric suffix
    step_dirs.sort(key=lambda p: int(p.name.split("_")[-1]))
    return step_dirs[-1]


def main():
    parser = argparse.ArgumentParser(description="Convert FrozenLake RL trajectories to LLM SFT chat JSONL")
    parser.add_argument("run_dir", help="Path to the run directory (contains trajectories/)" )
    parser.add_argument("--step", default=None, help="Specific step directory name (e.g., step_993280)")
    parser.add_argument("--include_failed", action="store_true", help="Include failed episodes in SFT data")
    parser.add_argument("--max_actions", type=int, default=25, help="Max actions cap for filtering and counter display")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir)
    traj_root = run_dir / "trajectories"
    if not traj_root.exists():
        raise FileNotFoundError(f"Not found trajectories directory: {traj_root}")

    step_dir = traj_root / args.step if args.step else find_latest_step_dir(traj_root)
    output_dir = run_dir / "sft"
    out_path = convert_file(step_dir=step_dir, output_dir=output_dir, repo_root=repo_root, include_failed=args.include_failed, max_actions=args.max_actions)
    print(f"SFT data written to: {out_path}")


if __name__ == "__main__":
    main()

