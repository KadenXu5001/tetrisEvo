import pickle
import neat
from tqdm import tqdm
import gymnasium as gym
import numpy as np
from tetris_gymnasium.envs import Tetris

def count_holes(board):
    holes = 0
    rows, cols = board.shape

    for col in range(cols):
        block_found = False  
        for row in range(rows):
            if board[row, col] > 0:
                block_found = True
            elif board[row, col] == 0 and block_found:
                holes += 1
    return holes


def bottom_row_fullness(board):
    bottom_row = board[-1, :]
    filled = np.count_nonzero(bottom_row)
    return filled / bottom_row.size


def extract_features(obs):
    board = obs["board"]
    rows, cols = board.shape

    # Column heights
    col_heights = np.array([
        rows - np.argmax(board[:, c] > 0) if np.any(board[:, c] > 0) else 0
        for c in range(cols)
    ], dtype=np.float32)

    # Holes per column
    holes = np.array([
        sum(1 for r in range(rows) if board[r, c] == 0 and np.any(board[:r, c] > 0))
        for c in range(cols)
    ], dtype=np.float32)

    aggregate_height = np.sum(col_heights)
    bumpiness = np.sum(np.abs(np.diff(col_heights)))

    # Active piece
    active = obs["active_tetromino_mask"].flatten()
    active_hot = active[:7]  # keep 7 inputs

    # Queue piece
    queue_hot = obs["queue"].flatten()[:7]

    # Held piece
    holder_hot = obs["holder"].flatten()[:7]

    # Return consistent 59-length vector (your config requires exactly 59 inputs)
    return np.concatenate([
        col_heights,            # 18
        holes,                  # 18
        [aggregate_height],     # 1
        [bumpiness],            # 1
        active_hot,             # 7
        queue_hot,              # 7
        holder_hot              # 7
    ]).astype(np.float32)


def board_metrics(board):
    rows, cols = board.shape

    heights = np.zeros(cols)
    holes = 0

    for c in range(cols):
        column = board[:, c]
        filled = np.where(column > 0)[0]
        if len(filled) > 0:
            heights[c] = rows - filled[0]
            holes += np.sum(column[filled[0]:] == 0)

    aggregate_height = np.sum(heights)
    bumpiness = np.sum(np.abs(np.diff(heights)))

    return aggregate_height, holes, bumpiness


def eval_genomes(genomes, config):
    env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
    board_height = 20  

    for genome_id, genome in tqdm(genomes):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0.0

        obs, info = env.reset(seed=42)
        prev_board = obs["board"].copy()
        terminated = truncated = False

        while not (terminated or truncated):
            inputs = extract_features(obs)
            output = np.nan_to_num(net.activate(inputs))
            action = int(np.clip(np.argmax(output), 0, 7))

            obs, reward, terminated, truncated, info = env.step(action)
        
            # ---- survival shaping ----
            fitness += 0.01


            board = obs["board"]

            prev_filled = np.sum(prev_board == 0)
            curr_filled = np.sum(board == 0 )


            piece_placed = curr_filled != prev_filled or info.get("lines_cleared", 0) > 0

            
            if piece_placed:
                
                lines = info.get("lines_cleared", 0)
                fitness += lines * 50.0
                
                if(lines > 3):
                    fitness += 400.0

                # reward bottom-heavy placement
                for y in range(board_height):
                    row_filled = np.sum(board[y] > 1)  # pieces only
                    height_weight = (y + 1) / board_height
                    fitness += row_filled * height_weight * 0.2

                
            
                placed_mask = (board > 1) & (prev_board <= 1)

                board_height, board_width = board.shape
                top_ignore_rows = 2
                middle_cols_start = board_width // 2 - 2  
                middle_cols_end = middle_cols_start + 4   

                ignore_mask = np.zeros_like(board, dtype=bool)

                # Only ignore the top 2 rows, middle 4 columns
                ignore_mask[:top_ignore_rows, middle_cols_start:middle_cols_end] = True

                # Final placed mask used for support calculation
                placed_mask_filtered = placed_mask & (~ignore_mask)
                
                totalNewHoles = 0
                unsupported = 0
                supported = 0
                for y in range(board_height):
                    row = ""
                    for x in range(board.shape[1]):
                        if placed_mask[y, x]:
                            piece_rows = np.where(placed_mask_filtered[:, x])[0]
                            if len(piece_rows) == 0:
                                row += "X"  
                                continue

                            lowest_y = piece_rows.max()
                            if y == lowest_y:
                                if board[y + 1, x] != 0:
                                    row += "S"  # supported
                                    supported += 1
                                    
                                else:
                                    row += "U"  # unsupported
                                    unsupported += 1
                                    
                            else:
                                row += "X"  
                            currentLowest = lowest_y
                            while board[currentLowest, x ] != 1:
                                if(board[currentLowest, x ] == 0):
                                    totalNewHoles += 1
                                currentLowest += 1
                        elif board[y, x] != 0:
                            row += "O"  
                        else:
                            row += "."

                supported_percent = supported / (supported + unsupported + 1e-5)
                fitness += supported_percent * 1 - (1-supported_percent) * 1.5
                
                if supported > 0 and unsupported == 0:
                    fitness += 6.0

                fitness -= min(totalNewHoles * 2, 40)/2


                prev_board = board.copy()

        if terminated:
            fitness -= 2.0

        genome.fitness = fitness

    env.close()


def run(resume=False):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-tetris-neat.ini"
    )

    if resume:
        with open("checkpoint_gen_8.pkl", "rb") as f:
            population = pickle.load(f)
    else:
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.StatisticsReporter())

    N_GENERATIONS = 50
    SAVE_INTERVAL = 4

    winner = population.best_genome
    with open("winner_rotate.pkl", "wb") as f:
        pickle.dump(winner, f)
    previousScore = -500
    current_best = winner
    for gen in range(N_GENERATIONS):
        print(f"\n--- Generation {gen} ---")
        
        population.run(eval_genomes, 1)
        score = population.best_genome.fitness
        if(score > previousScore):
            previousScore = score
            current_best = population.best_genome


        if gen % SAVE_INTERVAL == 0:
            with open(f"checkpoint_gen_{gen}.pkl", "wb") as f:
                pickle.dump(population, f)
            with open("best_genome.pkl", "wb") as f:
                pickle.dump(current_best, f)
            print(f"Saved checkpoint for generation {gen}")

    winner = population.best_genome
    print("Winner:", winner)

    with open("population_final.pkl", "wb") as f:
        pickle.dump(population, f)
    with open("winner_final.pkl", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    run(resume=True)
