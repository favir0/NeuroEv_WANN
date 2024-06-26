import numpy as np

from ff_nn import NeuralNetwork
from tasks.nn_task import NNTask


class LogicalOperationsTask(NNTask):
    def __init__(self, task: str = "XOR"):
        self.task = task
        self.epsilon = 0.1

        if task == "XOR":
            self.inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
            self.expected_outputs = [[0], [1], [1], [0]]
        elif task == "AND":
            self.inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
            self.expected_outputs = [[0], [0], [0], [1]]
        elif task == "OR":
            self.inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
            self.expected_outputs = [[0], [1], [1], [1]]
        else:
            raise ValueError(f"Task {task} is not supported yet.")


        self._name = "LogicalOperations"
        self._input_nodes = 2
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
        total_fitness = 0
        for input_vector, expected_output in zip(self.inputs, self.expected_outputs):
            predicted_output = neural_network.feed(input_vector)

            error = np.sum(np.abs(np.array(predicted_output) - np.array(expected_output)))

            error = error**2

            fitness = 1 / (1 + error)
            total_fitness += fitness
        return total_fitness / len(self.inputs)

    def solve(self, neural_network: NeuralNetwork) -> bool:
        for input_vector, expected_output in zip(self.inputs, self.expected_outputs):
            predicted_output = neural_network.feed(input_vector)
            if not all(np.abs(np.array(predicted_output) - np.array(expected_output)) < 0.5):
                return False
        return True

    def visualize(self, neural_network: NeuralNetwork):
        for input_vector, expected_output in zip(self.inputs, self.expected_outputs):
            predicted_output = neural_network.feed(input_vector)
            print(f"Input vector: {input_vector}")
            print(f"Expected output: {expected_output}")
            print(f"Pedicted output: {predicted_output}")
            print(f"Predicted output rounded: {np.round(predicted_output)}")
