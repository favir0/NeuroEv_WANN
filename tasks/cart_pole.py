import numpy as np
import gymnasium as gym

from ff_nn import NeuralNetwork
from tasks.nn_task import NNTask


class CartPoleTask(NNTask):
    def __init__(self, threshold: int = 5000):
        self.threshold = threshold
        self.episodes = 10
        self.epsilon = 0.1
        self.env = gym.make("CartPole-v1", max_episode_steps=5000)
        self.env.reset(seed=42)

    def evaluate(self, neural_network: NeuralNetwork) -> float:
        total_reward = 0
        for i in range(self.episodes):
            state, _ = self.env.reset()
            done = False
            truncate = False
            while not done and not truncate:
                output = neural_network.feed(state)
                action = 0 if output[0] < 0 else 1
                state, reward, done, truncate, _ = self.env.step(action)
                total_reward += reward
        average_reward = total_reward / self.episodes
        return average_reward

    def solve(self, neural_network: NeuralNetwork) -> bool:
        average_reward = self.evaluate(neural_network)
        return average_reward >= self.threshold

    def visualize(self, neural_network: NeuralNetwork):
        self.env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=5000)
        state, _ = self.env.reset()
        done = False
        truncate = False
        reward_total = 0
        while not done and not truncate:
            self.env.render()
            output = neural_network.feed(state)
            action = 0 if output[0] < 0 else 1
            state, reward, done, truncate, info = self.env.step(action)
            reward_total += reward
        self.env.close()
