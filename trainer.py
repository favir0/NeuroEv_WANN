from ff_nn import NeuralNetwork
from population import Population
from config import config
from wann_core import Genome
from tasks import (
    LogicalOperationsTask,
    BreastCancerTask,
    GlassTask,
    CartPoleTask,
    LunarLanderTask,
    BipedalWalkerTask
)

def save_genome_to_nn(genome: Genome, name_addition: str = ""):
    test_nn = NeuralNetwork(genome)
    if config.wann_use_weights_pool:
        test_nn.best_weight = genome.best_weight
        test_nn.weights_pool = config.wann_weights_pool
    test_nn.save(f"./outputs/{task_name}/{generation}{name_addition}")


if __name__ == "__main__":
    population = Population(evaluator=CartPoleTask)
    task_name = population.evaluator.task_name
    generations = 300
    for generation in range(generations):
        population.evolve()
        print(f"[{generation}] Champion fitness: {population.champions[-1].fitness}; Species: {len(population.species)}")
        if generation % 10 == 0:
            save_genome_to_nn(population.champions[-1])
        
        if population.solved_at is not None or generation == generations - 1:
            champion = population.champions[-1]
            print(f"Solved at {population.solved_at}")
            print(f"Champion fitness {champion.fitness}")
            save_genome_to_nn(champion, "_solved")
            
            champion.print_genome()
            test_nn = NeuralNetwork(champion)
            test_nn.visualize(show_weights=True)
            population.evaluator.visualize(test_nn)
            break
