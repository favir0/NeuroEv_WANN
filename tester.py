from ff_nn import NeuralNetwork
from population import Population
from tasks import (
    LogicalOperationsTask,
    BreastCancerTask,
    GlassTask,
    CartPoleTask,
    LunarLanderTask,
    BipedalWalkerTask
)

if __name__ == "__main__":
    population = Population(evaluator=BreastCancerTask)
    task_name = population.evaluator.task_name
    loaded_nn = NeuralNetwork.load(f"./outputs/{task_name}/90")
    loaded_nn.visualize(show_weights=True)
    population.evaluator.visualize(loaded_nn)
