import gymnasium as gym
import numpy as np
import pygame
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os

class CustomLunarLander(gym.Env):
    def __init__(self, render_mode=None):
        self.base_env = gym.make("LunarLander-v3", render_mode=render_mode)
        self.observation_space = gym.spaces.Box(
            low=np.append(self.base_env.observation_space.low, [-np.inf, -1.0, -1.0, -1.0]),
            high=np.append(self.base_env.observation_space.high, [np.inf, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        self.action_space = self.base_env.action_space
        self.fuel = 1000.0
        self.wind = np.random.uniform(-1.0, 1.0)
        self.checkpoint = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(0.3, 0.7)])

    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self.fuel = 1000.0
        self.wind = np.random.uniform(-1.0, 1.0)
        self.checkpoint = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(0.3, 0.7)])
        return self._get_obs(obs), info

    def step(self, action):
        # Apply wind by shifting lander manually
        if hasattr(self.base_env, 'lander') and self.base_env.lander:
            self.base_env.lander.ApplyForceToCenter((self.wind, 0), True)

        # Fuel consumption logic
        if self.fuel > 0:
            if action == 2:  # Main engine
                self.fuel -= 1.5
            elif action in [1, 3]:  # Side engines
                self.fuel -= 0.75
        else:
            action = 0  # No fuel

        obs, reward, terminated, truncated, info = self.base_env.step(action)

        # Reward shaping: approach checkpoint
        dist = np.linalg.norm(np.array(self.unwrapped.lander.position) - self.checkpoint)

        reward += 0.1 * (1.0 - dist)

        # Fuel penalty
        reward -= (1000.0 - self.fuel) * 0.0005

        return self._get_obs(obs), reward, terminated, truncated, info

    def _get_obs(self, obs):
        return np.append(obs, [self.fuel / 1000.0, self.wind, *self.checkpoint])

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self):
        if "episode" in self.locals["infos"][0]:
            ep_rew = self.locals["infos"][0]["episode"]["r"]
            self.rewards.append(ep_rew)
            avg_rew = np.mean(self.rewards[-50:])
            print(f"Episode: {len(self.rewards)}, Reward: {ep_rew:.2f}, Avg: {avg_rew:.2f}")
        return True

    def plot_rewards(self):
        plt.plot(self.rewards)
        plt.title("Reward over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid()
        plt.show()

def train_model():
    env = DummyVecEnv([lambda: CustomLunarLander()])
    model = DQN("MlpPolicy", env, verbose=0, tensorboard_log="./tensorboard/")
    callback = RewardLogger()
    model.learn(total_timesteps=100_000, callback=callback)
    model.save("lander_dqn_model")
    callback.plot_rewards()

def evaluate_model():
    env = CustomLunarLander(render_mode="human")
    model = DQN.load("lander_dqn_model", env=DummyVecEnv([lambda: CustomLunarLander()]))
    obs, _ = env.reset()
    total_reward = 0

    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Display info
        print(f"Reward: {reward:.2f}, Fuel: {obs[8]*1000:.1f}, Wind: {obs[9]:.2f}")

        if terminated or truncated:
            print(f"Episode done. Total reward: {total_reward:.2f}")
            total_reward = 0
            obs, _ = env.reset()

if __name__ == "__main__":
    mode = input("Enter 'train' to train or 'eval' to evaluate: ").strip().lower()
    if mode == "train":
        train_model()
    elif mode == "eval":
        evaluate_model()
    else:
        print("Invalid input. Please enter 'train' or 'eval'.")
