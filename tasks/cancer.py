import numpy as np

from ff_nn import NeuralNetwork
from tasks.nn_task import NNTask


def init_dataset():
    file_path = "data/cancer/cancer.raw"
    data_input = []
    data_output = []
    with open(file_path, "r") as file:
        for line in file:
            values = line.strip().split(",")
            try:
                v = [int(value)/10 for value in values[1:-1]]
                answers = [0 if int(values[-1]) == 2 else 1]
            except ValueError:
                continue
            data_input.append(v)
            data_output.append([answers])
    return data_input, data_output

def mse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2).mean()

class BreastCancerTask(NNTask):
    def __init__(self):
        self.tolerance = 0.95
        data_input, data_output = init_dataset()
        self.data_input = data_input
        self.data_output = data_output
        self.train_data = data_input[:400]
        self.test_data = data_input[400:]
        self.train_answers = data_output[:400]
        self.test_answers = data_output[400:]

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
        for input_vector, expected_output in zip(self.test_data, self.test_answers):
            predicted_output = neural_network.feed(input_vector)

            if all(
                np.abs(np.array(predicted_output) - np.array(expected_output)) < 0.5
            ):
                right_ans += 1
            else:
                wrong_ans += 1

        print(f"right answers: {right_ans}; wrong answers: {wrong_ans}")
