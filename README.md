import gymnasium as gym
import numpy as np
import pygame
import matplotlib.pyplot as plt
from gymnasium.envs.box2d import LunarLander
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

CHECKPOINT = np.array([0.5, 0.5])  # normalized position

class CustomLunarLander(LunarLander):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        high = np.array([np.inf] * 12, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.wind_power = 15.0
        self.fuel = 1000

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.fuel = 1000
        return self._get_obs(), info

    def step(self, action):
        if self.fuel <= 0:
            action = 0
        if action == 2:
            self.fuel -= 1  # main engine
        elif action in [1, 3]:
            self.fuel -= 0.5  # side engines

        obs, reward, done, truncated, info = super().step(action)
        distance_to_checkpoint = np.linalg.norm(np.array(self.lander.position) / 20 - CHECKPOINT)
        reward += (1.0 - distance_to_checkpoint)
        reward -= 0.001 * (1000 - self.fuel)

        if distance_to_checkpoint < 0.1:
            reward += 100
            done = True
            info["success"] = True

        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        pos = np.array(self.lander.position) / 20
        vel = np.array(self.lander.linearVelocity) / 20
        angle = np.array([self.lander.angle])
        angular_velocity = np.array([self.lander.angularVelocity])
        legs = [1.0 if l.ground_contact else 0.0 for l in self.legs]
        wind = np.array([self.wind_power])
        fuel = np.array([self.fuel / 1000])
        checkpoint = CHECKPOINT
        return np.concatenate([pos, vel, angle, angular_velocity, legs, wind, fuel, checkpoint])

# Reward logger
class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        if self.locals.get("rewards") is not None:
            self.rewards.append(self.locals["rewards"][0])
        return True

    def plot(self):
        plt.plot(self.rewards)
        plt.title("Reward Over Time")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.show()

def train_model():
    env = DummyVecEnv([lambda: CustomLunarLander()])
    model = DQN("MlpPolicy", env, verbose=1)
    reward_logger = RewardLogger()
    model.learn(total_timesteps=100_000, callback=reward_logger)
    model.save("dqn_lunar_custom")
    reward_logger.plot()

def evaluate_model():
    env = CustomLunarLander(render_mode="human")
    model = DQN.load("dqn_lunar_custom")
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    mode = input("Enter 'train' to train or 'eval' to evaluate: ").strip().lower()
    if mode == "train":
        train_model()
    elif mode == "eval":
        evaluate_model()
    else:
        print("Invalid input. Enter 'train' or 'eval'.")
