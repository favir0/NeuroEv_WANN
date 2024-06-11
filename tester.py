from ff_nn import NeuralNetwork
from population import Population
from tasks import (
    LogicalOperationsTask,
)

if __name__ == "__main__":
    population = Population(evaluator=LogicalOperationsTask)
    loaded_nn = NeuralNetwork.load("./outputs/LunarLander/57_solved")
    loaded_nn.visualize(show_weights=True)
    population.evaluator.visualize(loaded_nn)
