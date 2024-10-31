# RL-TP4 - Lyes Benacer - Ilyas Oulkadda

# Context

Based on the research from the paper ["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.5602), our objective is to implement a reinforcement learning agent using Deep Q-Learning (DQN) capable of playing Pong. This foundational work demonstrates how deep Q-networks can effectively learn to play Atari games, including Pong, using a convolutional neural network as a Q-function approximator. Our implementation follows a similar approach, leveraging the Arcade Learning Environment (ALE) to provide a framework for Atari game interactions and support for training DQN agents.

# Repository Architecture for Pong AI

This repository contains files and assets related to developing a deep Q-learning agent to play the game Pong. Below is a breakdown of the key components:

1. **`dqn_pong.py`**: This script implements the code to train the AI agent using a Deep Q-Network (DQN) to play Pong. The training process optimizes the agent’s actions to maximize its cumulative reward over time, allowing it to learn effective strategies to win the game.

2. **`play_pong.py`**: This script allows the agent to play Pong in its initial, untrained state. It serves as a native implementation of the agent, providing insight into its behavior before any learning takes place.

3. **Video of the AI Before Training**: This video captures the agent's performance in its untrained state. Watching this can help illustrate how the agent behaves when it has no experience or strategy for playing Pong.

4. **Video of the AI During/After Training**: This video shows the agent’s gameplay either mid-training or after training has completed. It provides a comparison point to the initial untrained state, highlighting the progress the AI has made in learning effective strategies.

5. **Reward Graph**: This graph visualizes the rewards received by the AI over time during training. It shows how the AI’s performance improves as it gains experience, with increasing rewards indicating successful learning and adaptation in gameplay.

6.  **dqn_pong_model.keras**: The weights of our trained neural networks

This setup allows you to both develop and visualize the agent's learning process, offering a comprehensive view of its performance from start to finish.


# Structure

The provided code implements a reinforcement learning (RL) framework to train a Deep Q-Network (DQN) agent capable of playing the Pong game. The structure includes the following key components:

1. **Preprocessing**:
   - The `preprocess_frame` function converts game frames to grayscale, resizes and crops them, and normalizes pixel values. This step ensures the input frames are formatted for effective training of the neural network.

2. **Frame Stacking**:
   - The `FrameStack` class holds a sequence of recent frames (default of four) to capture temporal context, essential for the DQN to recognize ball and paddle dynamics. It also provides functions for identifying the positions of the paddle and ball to aid reward shaping.

3. **Reward Shaping**:
   - In addition to Pong’s default rewards, a custom reward function `calculate_distance_reward` is used to encourage the agent to align the paddle with the ball. This shaping helps the agent learn fundamental positioning before tackling the scoring objectives in the game.

4. **Replay Memory**:
   - The `ReplayMemory` class enables experience replay by storing state transitions (state, action, reward, next state). By sampling from this stored memory, the DQN can learn from past experiences, enhancing the stability and efficiency of the training process.

5. **DQN Model**:
   - The `create_dqn_model` function defines a convolutional neural network architecture, tailored for processing visual inputs from the Pong game. This network processes states and outputs Q-values, which are used to select optimal actions.

6. **Training and Testing Loop**:
   - The `train_dqn` function manages the main training loop. It performs steps including action selection, memory storage, and network updating. After training, `test_game` enables the agent to play Pong, testing its learned policy.
  

# Implementation Choices

When training our DQN agent for Pong, we encountered a significant challenge: the original Pong reward structure only provides:
  - +1 for scoring a point,
  - -1 for losing a point,
  - and 0 for most other actions.

At the start, with an epsilon value of 1, the agent begins by taking completely random actions. This setup makes it unlikely to earn positive rewards, leading to a slow learning process as the agent struggles to associate actions with positive outcomes.

### Solution: Reward Shaping Based on Paddle and Ball Position

To address this, we implemented *reward shaping* focused on the *y-axis distance between the paddle and the ball*. This approach provides immediate feedback on paddle positioning, which helps the agent develop skills crucial for consistently hitting the ball. Here’s how our reward shaping is structured:

1. **Perfect Alignment Reward**: If the paddle is at the same height as the ball, the agent receives a reward of **+3**, reinforcing the goal of keeping the paddle in line with the ball.

2. **Close Alignment Reward**: If the paddle is near the ball but not perfectly aligned, the agent earns a smaller reward of **+0.5**. This teaches the agent that proximity to the ball is still beneficial, even if it's not in perfect alignment.

3. **Penalty for Misalignment**: If the paddle is far from the ball, we apply a penalty of **-0.5** to discourage poor positioning.

This reward structure gives the agent feedback on almost every frame, helping it learn positioning skills early in training. Once the agent reliably aligns the paddle with the ball, we can return to the original Pong reward system based solely on points, allowing it to focus on scoring and refining its strategies.

# Results

We decided to stop the training after 50 episodes because it was taking too much time. Initially, our model struggled significantly, losing 21-0 in many games. However, after 51 games, the performance improved noticeably, with losses reduced to 9-21. This change indicates that the agent can return the ball more consistently, although it still struggles to do so every time, leading to rapid distancing from the opponent.

While we believe further training could enhance the model's performance, we also think optimizing the neural network structure, refining the reward function, or adjusting how we send the environment to the neural network (currently using 4 stacked frames) may yield better results.




