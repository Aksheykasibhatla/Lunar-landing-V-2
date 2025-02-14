import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


env = gym.make("LunarLander-v3", render_mode="human")
env = Monitor(env)  # 🔥 This fixes the warning!


env = DummyVecEnv([lambda: env])


model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)  


model.save("lunar_lander_a2c")


model = A2C.load("lunar_lander_a2c")


mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")


obs = env.reset()
for _ in range(1000):  # Run for more steps
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, _ = env.step(action)
    env.render()

env.close()








