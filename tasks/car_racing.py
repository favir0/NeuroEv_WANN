import numpy as np
import gymnasium as gym

from ff_nn import NeuralNetwork
from tasks.nn_task import NNTask


class CarRacingTask(NNTask):
    def __init__(self, tolerance: float = 0.1, error_type: str = "mse"):
        self.tolerance = tolerance
        self.error_type = error_type
        self.epsilon = 0.1
        self.env = gym.make("CarRacing-v2", render_mode="human")
        self.env.reset(seed=42)

    def evaluate(self, neural_network: NeuralNetwork, episodes: int = 10) -> float:
        total_reward = 0
        for i in range(episodes):
            state, _ = self.env.reset()
            done = False
            steps = 0
            while not done and steps < 1000:
                action = self.get_action(neural_network, state)
                state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                steps += 1
        average_reward = total_reward / episodes
        return average_reward

    def solve(self,neural_network: NeuralNetwork,threshold: float = 900.0,episodes: int = 10,) -> bool:
        average_reward = self.evaluate(neural_network, episodes)
        return average_reward >= threshold

    def get_action(self, neural_network: NeuralNetwork, state: np.ndarray) -> np.ndarray:
        output = neural_network.feed(state)
        action = np.clip(output, -1, 1)
        return action

    def visualize(self, neural_network: NeuralNetwork, episodes: int = 1):
        self.env = gym.make("CarRacing-v2", render_mode="human")
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                self.env.render()
                action = self.get_action(neural_network, state)
                state, reward, done, _, info = self.env.step(action)
        self.env.close()
