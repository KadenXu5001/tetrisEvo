import numpy as np
import cma
import os
from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations

SAVE_FILE = "best_weights.npy"

# -----------------------------
# Board metrics / heuristic features
# -----------------------------
def board_metrics(board):
    rows, cols = board.shape
    heights = np.zeros(cols)
    holes = 0
    row_transitions = 0
    col_transitions = 0
    wells = 0

    # Heights + holes
    for c in range(cols):
        col = board[:, c]
        filled = np.where(col > 0)[0]
        if len(filled) > 0:
            heights[c] = rows - filled[0]
            holes += np.sum(col[filled[0]:] == 0)

    # Row transitions
    for r in range(rows):
        row = board[r]
        prev = 1
        for cell in row:
            if cell != prev:
                row_transitions += 1
            prev = cell
        if prev == 0:
            row_transitions += 1

    # Column transitions
    for c in range(cols):
        prev = 1
        for r in range(rows):
            cell = board[r, c]
            if cell != prev:
                col_transitions += 1
            prev = cell
        if prev == 0:
            col_transitions += 1

    # Wells
    for c in range(cols):
        left = heights[c - 1] if c > 0 else rows
        right = heights[c + 1] if c < cols - 1 else rows
        min_neighbor = min(left, right)
        if min_neighbor > heights[c]:
            wells += min_neighbor - heights[c]

    aggregate_height = np.sum(heights)
    bumpiness = np.sum(np.abs(np.diff(heights)))
    max_height = np.max(heights)

    return np.array([
        aggregate_height,
        holes,
        bumpiness,
        max_height,
        row_transitions,
        col_transitions,
        wells,
    ], dtype=np.float32)

def evaluate_board(board, weights):
    features = board_metrics(board)
    return np.dot(weights, features)

# -----------------------------
# NumPy Tetris lookahead simulator
# -----------------------------
def simulate_action(board, piece, col, rotation):
    """
    Place a tetromino at a column with rotation on a copy of the board.
    Returns new board after placement and lines cleared.
    """
    board = board.copy()
    # The Tetris-Gymnasium env has pieces encoded 0-6
    # For simplicity we just drop the piece in the first empty row
    # Note: this is a simplified simulation, works for CMA-ES heuristic
    rows, cols = board.shape
    piece_height = 4  # assume max 4x4 piece mask
    # Drop piece
    for r in range(rows - piece_height + 1):
        if np.any(board[r:r+piece_height, col:col+4] != 0):
            r -= 1
            break
    r = max(r, 0)
    board[r:r+piece_height, col:col+4] = piece + 1  # simplified piece placement
    # Count cleared lines
    lines_cleared = 0
    full_rows = np.all(board > 0, axis=1)
    lines_cleared = np.sum(full_rows)
    board[full_rows] = 0
    return board, lines_cleared

# -----------------------------
# Choose best action with lookahead
# -----------------------------
def choose_best_action(board, active_piece, info, weights):
    """
    board: 2D array copy
    active_piece: int (current tetromino)
    info: dict from grouped wrapper (must have 'action_mask')
    weights: heuristic weights
    """
    legal_actions = np.where(info["action_mask"])[0]  # only allowed grouped actions
    best_score = -1e12
    best_action = legal_actions[0]

    for action in legal_actions:
        # map grouped action to column + rotation
        col = action // 4
        rotation = action % 4
        sim_board, _ = simulate_action(board, active_piece, col, rotation)
        score = evaluate_board(sim_board, weights)

        if score > best_score:
            best_score = score
            best_action = action

    return best_action

# -----------------------------
# Fitness evaluation
# -----------------------------
def evaluate_weights(weights, seed=42, render=False):
    # Create environment with grouped actions
    env = Tetris(render_mode="human" if render else None)
    env = GroupedActionsObservations(env)

    obs, info = env.reset(seed=seed)

    # Initialize our own board copy for lookahead
    board = np.zeros((20, 10), dtype=int)  # Tetris standard size
    active_piece = 0  # placeholder; could extract from obs if needed

    terminated = truncated = False
    total_lines = 0
    steps = 0

    while not (terminated or truncated):
        # Pick the best action using your board copy for lookahead
        action = choose_best_action(board, active_piece, info, weights)

        # Take the action in the actual Gym environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_lines += info.get("lines_cleared", 0)
        steps += 1

        # Update our own board copy with the action
        board, lines_cleared = simulate_action(board, active_piece, 
                                               col=action // 4, 
                                               rotation=action % 4)
        total_lines += lines_cleared

        # Update active piece from the observation if possible
        # This depends on how your environment encodes the next piece
        active_piece = 0  # or extract from obs if available

        # Safety cap to prevent infinite games
        if steps > 5000:
            break

        # Optional rendering
        if render:
            env.render()

    env.close()
    return total_lines


# -----------------------------
# CMA-ES training loop
# -----------------------------
def train():
    dim = 7
    sigma = 1.0
    if os.path.exists(SAVE_FILE):
        x0 = np.load(SAVE_FILE)
        print("Loaded weights:", x0)
    else:
        x0 = np.zeros(dim)

    es = cma.CMAEvolutionStrategy(
        x0, sigma,
        {"popsize": 16, "seed": 0, "verb_disp": 1}
    )

    generation = 0
    while not es.stop() and generation < 50:
        solutions = es.ask()
        fitnesses = []

        for w in solutions:
            f = evaluate_weights(w, seed=42)
            fitnesses.append(-f)

        es.tell(solutions, fitnesses)
        es.disp()

        np.save(SAVE_FILE, es.result.xbest)
        generation += 1

    print("Best weights:", es.result.xbest)
    print("Best fitness:", -es.result.fbest)
    print("Playing with best weights...")
    evaluate_weights(es.result.xbest, seed=42, render=True)

# -----------------------------
if __name__ == "__main__":
    train()
