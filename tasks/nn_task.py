from ff_nn import NeuralNetwork
from abc import ABC, abstractmethod


class NNTask(ABC):
    @abstractmethod
    def evaluate(self, neural_network: NeuralNetwork) -> float:
        pass

    @abstractmethod
    def solve(self, neural_network: NeuralNetwork) -> bool:
        pass

    @abstractmethod
    def visualize(self, neural_network: NeuralNetwork):
        pass

    @property
    @abstractmethod
    def task_name(self) -> str:
        pass

    @property
    @abstractmethod
    def input_nodes(self) -> int:
        pass

    @property
    @abstractmethod
    def output_nodes(self) -> int:
        pass
