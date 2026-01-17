import gymnasium as gym
from tetris_gymnasium.envs import Tetris

env = gym.make("tetris_gymnasium/Tetris", render_mode=None)  
print(env.action_space)
observation, info = env.reset()
print("Observation space:", env.observation_space)
print("Observation shape:", observation.shape if hasattr(observation, 'shape') else None)
print("Action space:", env.action_space)
try:
    print("Number of discrete actions:", env.action_space.n)
except:
    pass
env.close()