import gymnasium as gym
import numpy as np

from ff_nn import NeuralNetwork
from tasks.nn_task import NNTask


class LunarLanderTask(NNTask):
    def __init__(self, threshold: int = 0.95):
        self.threshold = threshold
        self.episodes = 3
        self.env = gym.make("LunarLander-v2")
        self.env.reset(seed=100)
        self.min_reward = -300
        self.max_reward = 250

        self._name = "LunarLander"
        self._input_nodes = self.env.observation_space.shape[0]
        self._output_nodes = self.env.action_space.n
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


    def preprocess_state(self, state: np.ndarray) -> np.ndarray:
        high = self.env.observation_space.high
        low = self.env.observation_space.low
        state = 2 * (state - low) / (high - low) - 1
        return state

    def normalize_reward(self, reward: float) -> float:
        clamped_reward = np.clip(reward, self.min_reward, self.max_reward)
        normalized_reward = (clamped_reward - self.min_reward) / (self.max_reward - self.min_reward)
        return np.clip(normalized_reward, 0, 1)

    def evaluate(self, neural_network: NeuralNetwork) -> float:
        total_reward = 0
        for _ in range(self.episodes):
            state, _ = self.env.reset(seed=100)
            done = False
            truncate = False
            steps = 0
            while not done and not truncate:
                output = neural_network.feed(state)
                action = np.argmax(output)
                state, reward, done, truncate, _ = self.env.step(action)
                total_reward += reward
                steps += 1
        
        average_reward = self.normalize_reward(total_reward / self.episodes)
        return average_reward

    def idk_evaluate(self, neural_network: NeuralNetwork) -> float:
        total_reward = 0
        for _ in range(self.episodes):
            state, _ = self.env.reset(seed=100)
            done = False
            truncate = False
            steps = 0
            while not done and not truncate:
                output = neural_network.feed(state)
                action = np.argmax(output)
                state, reward, done, truncate, _ = self.env.step(action)
                #state = self.preprocess_state(state)
                total_reward += reward
                steps += 1
            average_reward = (total_reward / self.episodes)
        return average_reward

    def solve(self, neural_network: NeuralNetwork) -> bool:
        average_reward = self.evaluate(neural_network)
        print(self.idk_evaluate(neural_network))
        return average_reward >= self.threshold

    def visualize(self, neural_network: NeuralNetwork):
        env = gym.make("LunarLander-v2", render_mode="human")
        state, _ = env.reset(seed=100)
        done = False
        reward_total = 0
        truncate = False
        while not done and not truncate:
            env.render()
            output = neural_network.feed(state)
            action = np.argmax(output)
            state, reward, done, truncate, _ = env.step(action)
            reward_total += reward
        print(f"Total reward: {reward_total}")
        print(f"Normalized reward: {self.normalize_reward(reward_total)}")
        env.close()