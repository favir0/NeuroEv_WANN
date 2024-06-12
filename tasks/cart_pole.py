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

        self._name = "CartPole"
        self._input_nodes = self.env.observation_space.shape[0]
        self._output_nodes = 1
        print(f"Initialized '{self._name}' task with {self._input_nodes} inputs and {self._output_nodes} outputs")
    
    @property
    def task_name(self) -> str:
        return self._name

    @property
    def input_nodes(self) -> int:
        return self._input_nodes

    @property
    def output_nodes(self) -> int:
        return self._output_nodes

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
        env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=300)
        state, _ = env.reset()
        done = False
        truncate = False
        reward_total = 0
        while not done and not truncate:
            env.render()
            output = neural_network.feed(state)
            action = 0 if output[0] < 0 else 1
            state, reward, done, truncate, info = env.step(action)
            reward_total += reward
        env.close()
