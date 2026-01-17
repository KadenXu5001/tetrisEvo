import numpy as np
import gymnasium as gym
import cv2
from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
import os

SAVE_FILE = "best_weights.npy"


# -----------------------------
# Board heuristic features
# -----------------------------
def board_metrics(board):
    board = np.array(board, dtype=int)

    rows, cols = board.shape
    heights = np.zeros(cols)
    holes = 0

    for c in range(cols):
        col = board[:, c]
        filled = np.where(col > 0)[0]
        if len(filled) > 0:
            heights[c] = rows - filled[0]
            holes += np.sum(col[filled[0]:] == 0)

    aggregate_height = np.sum(heights)
    bumpiness = np.sum(np.abs(np.diff(heights)))
    max_height = np.max(heights)

    return np.array([
        aggregate_height,
        holes,
        bumpiness,
        max_height,
        0.0,  # padding to keep 7 features
        0.0,
        0.0,
    ], dtype=np.float32)


def evaluate_board(board, weights):
    return float(np.dot(weights, board_metrics(board)))


# -----------------------------
# Correct grouped-action chooser
# -----------------------------
def choose_best_action(obs, weights):
    """
    obs is produced by GroupedActionsObservations.
    obs["boards"] shape: (num_actions, 20, 10)
    """
    if not isinstance(obs, dict) or "boards" not in obs:
        raise RuntimeError("Grouped observation missing 'boards'")

    boards = obs["boards"]
    best_score = -1e12
    best_action = 0

    for action in range(len(boards)):
        board = boards[action]
        score = evaluate_board(board, weights)

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


# -----------------------------
# Play one game (visualized)
# -----------------------------
def play_one_game(weights, seed=42, render=True, delay=0.1, debug=True):
    env = Tetris(render_mode="human" if render else None)
    env = GroupedActionsObservations(env)

    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False
    steps = 0
    lines = 0

    if debug:
        print("--- Game Start ---")

    while not terminated and not truncated:
        try:
            action = choose_best_action(obs, weights)
        except Exception as e:
            print(f"Error in choose_best_action at step {steps}: {e}")
            break

        obs, reward, terminated, truncated, info = env.step(action)
        lines += info.get("lines_cleared", 0)
        steps += 1

        if render:
            env.render()
            cv2.waitKey(int(delay * 1000))

        if debug:
            print(f"Step {steps}, Lines: {lines}")

        if steps > 5000:
            print("Safety stop triggered")
            break

    env.close()
    return lines


# -----------------------------
# Play multiple games
# -----------------------------
def play_multiple_games(weights, num_games=1, delay=0.1):
    print("Starting agent playback...")

    for i in range(num_games):
        print(f"\n--- Game {i + 1} ---")
        lines = play_one_game(
            weights,
            seed=42 + i,
            render=True,
            delay=delay,
            debug=True,
        )
        print(f"Game finished! Lines cleared: {lines}")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    if not os.path.exists(SAVE_FILE):
        raise FileNotFoundError("No CMA-ES weights found")

    weights = np.load(SAVE_FILE)
    print("Loaded CMA-ES best weights:", weights)

    play_multiple_games(weights, num_games=1, delay=0.1)
