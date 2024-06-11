import numpy as np

from ff_nn import NeuralNetwork
from tasks.nn_task import NNTask


def init_dataset():
    file_path = "data/glass/glass1.dt"
    data_input = []
    data_output = []
    with open(file_path, "r") as file:
        for line in file:
            values = line.strip().split(" ")
            try:
                v = [float(value) for value in values[:9]]
                answers = [int(value) for value in values[9:]]
            except ValueError:
                continue
            data_input.append(v)
            data_output.append(answers)

    return data_input, data_output

def mse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2).mean()

class GlassTask(NNTask):
    def __init__(self):
        self.tolerance = 0.9
        data_input, data_output = init_dataset()
        self.data_input = data_input
        self.data_output = data_output
        self.train_data = data_input[:150]
        self.test_data = data_input[150:]
        self.train_answers = data_output[:150]
        self.test_answers = data_output[150:]

    def evaluate(self, neural_network: NeuralNetwork) -> float:
        total_fitness = 0
        for input_vector, expected_output in zip(self.train_data, self.train_answers):
            predicted_output = neural_network.feed(input_vector)
            error = mse(
                np.array(expected_output), np.array(predicted_output)
            )
            fitness = 1 / (1 + error)
            total_fitness += fitness

        return total_fitness / len(self.train_data)

    def solve(self, neural_network: NeuralNetwork) -> bool:
        return self.evaluate(neural_network) > self.tolerance

    def visualize(self, neural_network: NeuralNetwork):
        right_ans = 0
        wrong_ans = 0
        for input_vector, expected_output in zip(self.train_data, self.train_answers):
            predicted_output = neural_network.feed(input_vector)

            if np.argmax(predicted_output) == np.argmax(expected_output):
                right_ans += 1
            else:
                wrong_ans += 1

        print(f"right answers: {right_ans}; wrong answers: {wrong_ans}")
