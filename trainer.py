import pickle
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
    test_nn.save(f"./outputs/{task_name}/{generation}{name_addition}")


def save(population: Population, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(population, f)

if __name__ == "__main__":
    population = Population(evaluator=LunarLanderTask)
    task_name = population.evaluator.task_name
    generations = 300
    for generation in range(generations):
        population.evolve()
        print(f"[{generation}] Champion fitness: {population.champions[-1].fitness}; Species: {len(population.species)}")
        if generation % 10 == 0:
            save_genome_to_nn(population.champions[-1])
            save(population, f"./outputs/PopulationDumps/{task_name}_evo_pop_new_solution_{generation}")
        
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
    
    save(population, f"./outputs/PopulationDumps/{task_name}_evo_pop_new_solution_result")