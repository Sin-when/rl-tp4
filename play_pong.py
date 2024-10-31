import ale_py

import gymnasium as gym
import numpy as np


gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode="human")

observation, info = env.reset()


for _ in range(10000):  
    env.render()
    
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Check if the episode is done
    done = terminated or truncated
    
    if done:
        observation, info = env.reset()


env.close()
