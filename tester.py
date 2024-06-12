from ff_nn import NeuralNetwork
from population import Population
from config import config
from tasks import (
    LogicalOperationsTask,
    BreastCancerTask,
    GlassTask,
    CartPoleTask,
    LunarLanderTask,
    BipedalWalkerTask
)

test_on_different_weights = True

if __name__ == "__main__":
    population = Population(evaluator=BreastCancerTask)
    task_name = population.evaluator.task_name
    loaded_nn = NeuralNetwork.load(f"./outputs/{task_name}/26_solved")

    if test_on_different_weights:
        print("Different shared weights test")
        for weight in config.wann_weights_pool:
            if abs(weight - loaded_nn.genome.best_weight) < 0.001:
                print("Best weight result")
            loaded_nn.set_all_weights(weight)
            population.evaluator.evaluate(loaded_nn)
            loaded_nn.visualize(show_weights=True)
            population.evaluator.visualize(loaded_nn)
    
    else:
        print("Single shared weight test")
        if loaded_nn.genome.best_weight:
            loaded_nn.set_all_weights(loaded_nn.genome.best_weight)
        population.evaluator.evaluate(loaded_nn)
        loaded_nn.visualize(show_weights=True)
        population.evaluator.visualize(loaded_nn)
    

