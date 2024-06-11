from ff_nn import NeuralNetwork
from abc import ABC, abstractmethod


class NNTask(ABC):
    @abstractmethod
    def evaluate(self, neural_network: NeuralNetwork) -> float:
        pass

    @abstractmethod
    def solve(self, neural_network: NeuralNetwork) -> float:
        pass

    @abstractmethod
    def visualize(self, neural_network: NeuralNetwork):
        pass
