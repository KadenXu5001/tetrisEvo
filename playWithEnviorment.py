import sys
import numpy as np
import cv2
import gymnasium as gym

from tetris_gymnasium.envs import Tetris

if __name__ == "__main__":
    # Create an instance of Tetris
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    
    obs, info = env.reset(seed=42)
    prev_board = obs["board"].copy()

    # Main game loop
    terminated = False
    while not terminated:
        # Render the current state of the game as text
        env.render()

        # Pick an action from user input mapped to the keyboard
        action = None
        while action is None:
            key = cv2.waitKey(1)

            if key == ord("j"):
                action = env.unwrapped.actions.move_left
            elif key == ord(";"):
                action = env.unwrapped.actions.move_right
            elif key == ord("l"):
                action = env.unwrapped.actions.move_down
            elif key == ord("o"):
                action = env.unwrapped.actions.rotate_counterclockwise
            elif key == ord("z"):
                action = env.unwrapped.actions.rotate_clockwise
            elif key == ord(" "):
                action = env.unwrapped.actions.hard_drop
            elif key == ord("c"):
                action = env.unwrapped.actions.swap
            elif key == ord("r"):
                env.reset(seed=42)
                break

            if (
                cv2.getWindowProperty(env.unwrapped.window_name, cv2.WND_PROP_VISIBLE)
                == 0
            ):
                sys.exit()

        # Perform the action

        obs, reward, terminated, truncated, info = env.step(action)
        
        # ---- survival shaping ----
        fitness = 0

        board_height = 20

        board = obs["board"]

        prev_filled = np.sum(prev_board == 0)
        curr_filled = np.sum(board == 0 )


        piece_placed = curr_filled != prev_filled or info.get("lines_cleared", 0) > 0

        
        if piece_placed:
            
            lines = info.get("lines_cleared", 0)
            fitness += lines * 50.0
            
            if(lines > 3):
                fitness += 400.0

            placed_mask = (board != 0) & (prev_board == 0)

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
            
            unsupported = 0
            supported = 0
            totalNewHoles = 0
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
                        hole = 0
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

            print(f"New Holes: {totalNewHoles}")
            fitness -= totalNewHoles * 2
            
            if supported > 0 and unsupported == 0:
                fitness += 6.0


            prev_board = board.copy()


        



    # Game over
    print("Game Over!")