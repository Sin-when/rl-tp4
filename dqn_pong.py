import os
import random
import time
import ale_py
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from collections import deque, Counter
import matplotlib.pyplot as plt

# Suppress TensorFlow logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
tf.get_logger().setLevel('ERROR')  

# Configuration
ENV_NAME = 'ALE/Pong-v5'
MEMORY_SIZE = 100000
BATCH_SIZE = 32
GAMMA = 0.99
ALPHA = 0.00025
UPDATE_TARGET_FREQUENCY = 10000

def preprocess_frame(frame):
    # Convert to grayscale
    gray = tf.image.rgb_to_grayscale(frame)
    
    # Resize to 110x84
    resized = tf.image.resize(gray, [110, 84])
    
    # Crop to 84x84 (remove 18 rows from the top and 9 rows from the bottom)
    cropped = resized[18:-9, :]
    padded = tf.pad(cropped, [[0, 1], [0, 0], [0, 0]], constant_values=87)
    
    # Normalize and convert to a 2D numpy array
    normalized = tf.squeeze(padded) / 255.0
    
    return normalized.numpy()

class FrameStack:
    def __init__(self, initial_frame, size=4):
        self.size = size
        self.frames = deque([initial_frame] * size, maxlen=size)
        self.frame_dir = "initial_frame"
        os.makedirs(self.frame_dir, exist_ok=True)
        self.inc = 0

    def add_frame(self, frame):
        self.frames.append(frame)
      

    def get_state(self):
        return np.stack(self.frames, axis=-1)

    def get_ball_and_paddle_y(self):
        latest_frame = self.frames[-1]
        
        paddle_pixels = np.where((latest_frame[:, 74] * 255 == 147))
        if len(paddle_pixels[0]) > 0:
            paddle_y = np.mean(paddle_pixels[0])
        else:
            paddle_y = None  # Paddle not found

        ball_pixels = np.where(latest_frame * 255 >= 236) 
        if len(ball_pixels[0]) > 0:
            ball_y = np.mean(ball_pixels[0])
        else:
            ball_y = None  
  
        return ball_y, paddle_y

  

def calculate_distance_reward(curr_ball_y, curr_paddle_y):
    if None in (curr_ball_y, curr_paddle_y):
        return 0

    if curr_ball_y == curr_paddle_y:
        return 3
    elif abs(curr_ball_y - curr_paddle_y) <= 5:
        return 0.5
    else:
        return -0.5
    


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def create_dqn_model(num_actions):
    inputs = keras.layers.Input(shape=(84, 84, 4))
    
    x = keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    x = keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    outputs = keras.layers.Dense(num_actions)(x)

    return keras.Model(inputs=inputs, outputs=outputs)

class DQNAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.model = create_dqn_model(num_actions)
        self.target_model = create_dqn_model(num_actions)
        self.target_model.set_weights(self.model.get_weights())
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = 0.2
        self.optimizer = keras.optimizers.Adam(learning_rate=ALPHA)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)

        next_q_values = self.target_model.predict(next_states, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1)
        target_q_values = rewards + (1 - np.array(dones)) * GAMMA * max_next_q_values

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            one_hot_actions = tf.one_hot(actions, self.num_actions)
            q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            loss = keras.losses.MeanSquaredError()(target_q_values, q_values)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())




def train_dqn():
    gym.register_envs(ale_py)
    env = gym.make(ENV_NAME, render_mode="human")
    agent = DQNAgent(env.action_space.n)

    episode_rewards = []  # List to store rewards
    total_frames = 0
    
    for episode in range(50): 
        observation, info = env.reset()

        initial_frame = preprocess_frame(observation)
        frame_stack = FrameStack(initial_frame)
        current_state = frame_stack.get_state()

        episode_reward = 0
        terminated = truncated = False
        player_score = 0
        opponent_score = 0

        prev_ball_y, _ = frame_stack.get_ball_and_paddle_y()


        while not (terminated or truncated):
            action = agent.get_action(current_state)
            next_observation, reward, terminated, truncated, info = env.step(action)

            next_frame = preprocess_frame(next_observation)
            frame_stack.add_frame(next_frame)
            next_state = frame_stack.get_state()

            # Calculate reward shaping
            curr_ball_y, curr_paddle_y = frame_stack.get_ball_and_paddle_y()
            curr_ball_y = curr_ball_y if curr_ball_y is not None else prev_ball_y

            distance_reward = calculate_distance_reward(curr_ball_y, curr_paddle_y)
            shaped_reward = reward + distance_reward

            agent.memory.add(current_state, action, shaped_reward, next_state, terminated or truncated)
            agent.train()

            if total_frames % UPDATE_TARGET_FREQUENCY == 0:
                agent.update_target_model()

            current_state = next_state
            episode_reward += shaped_reward
            total_frames += 1

            # Track score based on the raw reward
            if reward == 1:
                player_score += 1
            elif reward == -1:
                opponent_score += 1

            prev_ball_y = curr_ball_y

        episode_rewards.append(episode_reward)  # Store the episode reward
        print(f"Episode {episode}, Game Score: {player_score}-{opponent_score}, Total Reward: {episode_reward}")

        # Decrease epsilon over time
        if episode == 20:
            agent.epsilon = 0.05

        if episode == 40:
            agent.epsilon = 0.01

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, 'b-')
    plt.title('Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('training_rewards.png')
    plt.close()

    # Save the model
    agent.model.save("dqn_pong_model.keras")

    # Make a simple Pong game with the model
    input("We have finished to train, Press ENTER to test the game")
    test_game(agent, env)


def test_game(agent, env):
    observation, info = env.reset()

    initial_frame = preprocess_frame(observation)
    frame_stack = FrameStack(initial_frame)
    current_state = frame_stack.get_state()

    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.get_action(current_state)
        next_observation, reward, terminated, truncated, info = env.step(action)

        next_frame = preprocess_frame(next_observation)
        frame_stack.add_frame(next_frame)
        next_state = frame_stack.get_state()

        current_state = next_state


def load_trained_model():
    env = gym.make(ENV_NAME, render_mode="human")
    agent = DQNAgent(env.action_space.n)
    
    # Load the saved weights
    try:
        agent.model.load_weights("dqn_pong_model.keras")
        print("Successfully loaded pre-trained model")
    except:
        print("No pre-trained model found")
        return None, None
        
    return agent, env

if __name__ == "__main__":
    train_dqn()