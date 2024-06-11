from ff_nn import NeuralNetwork
from population import Population
from tasks import (
    LogicalOperationsTask,
    BreastCancerTask,
    GlassTask,
    CartPoleTask,
    LunarLanderTask,
    CarRacingTask,
    BipedalWalkerTask
)

if __name__ == "__main__":
    population = Population(evaluator=CartPoleTask)
    generations = 300
    for generation in range(generations):  # количество поколений
        population.evolve()
        print(f"[{generation}] Champion fitness: {population.champions[-1].fitness}; Species: {len(population.species)}")
        if (generation % 10 == 0):
            test_nn = NeuralNetwork(population.champions[-1])
            test_nn.save(f"./outputs/CartPole/{generation}")
        
        if population.solved_at is not None or generation == generations - 1:
            print(f"Solved at {population.solved_at}")
            print(f"Champion fitness {population.champions[-1].fitness}")
            test_nn = NeuralNetwork(population.champions[-1])
            test_nn.save(f"./outputs/CartPole/{generation}_solved")
            population.champions[-1].print_genome()
            test_nn.visualize(show_weights=True)
            population.evaluator.visualize(test_nn)
            break