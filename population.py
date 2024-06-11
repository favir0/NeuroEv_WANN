import random

from copy import deepcopy
from config import config
from wann_core import Genome, genome_distance, genome_crossover, innov_tracker
from ff_nn import NeuralNetwork
from species import Species
from tasks.nn_task import NNTask


def tournament_selection(genomes: list[Genome], k=3):
    selected = random.sample(genomes, k)
    selected.sort(key=lambda genome: genome.fitness, reverse=True)
    return selected[0]


def create_initial_genome() -> Genome:
    individual = Genome()

    input_nodes = []
    for _ in range(config.input_nodes):
        node = individual.add_node(0)
        input_nodes.append(node)

    if config.add_bias_node:
        node = individual.add_node(0)

    out_nodes = []
    for _ in range(config.output_nodes):
        node = individual.add_node(config.max_depth + 1)
        out_nodes.append(node)

    for input_node in input_nodes:
        if random.random() < config.initial_connection_prob:
            out_node = random.choice(out_nodes)
            weight = config.wann_initial_weight
            individual.add_connection(input_node, out_node, weight)
    return individual


class Population:
    def __init__(self, evaluator):
        self.genomes: list[Genome] = [create_initial_genome() for _ in range(config.population_size)]
        self.species: list[Species] = []
        self.current_compatibility_threshold: int = config.compatibility_threshold
        self.champions: list[Genome] = []
        self.evaluator: NNTask = evaluator()
        self.solved_at: int | None = None
        self.generation_index: int = 0

    def speciate(self):
        for specie in self.species:
            specie.evolve_step()

        for genome in self.genomes:
            for specie in self.species:
                if (genome_distance(genome, specie.representative) < self.current_compatibility_threshold):
                    specie.add_genome(genome)
                    break
            else:
                new_specie = Species(representative=deepcopy(genome))
                new_specie.add_genome(genome)
                self.species.append(new_specie)

        
        self.species = list(filter(lambda s: len(s.genomes) > 0, self.species))

        if len(self.species) < config.target_species:
            self.current_compatibility_threshold -= config.compatibility_threshold_delta
        elif len(self.species) > config.target_species:
            self.current_compatibility_threshold += config.compatibility_threshold_delta
        if self.current_compatibility_threshold < config.min_compatibility_threshold:
            self.current_compatibility_threshold = config.min_compatibility_threshold

    def evaluate_all_fitness(self):
        for genome in self.genomes:
            genome.fitness = self.evaluate_genome(genome)
        for specie in self.species:
            specie.full_recalculate()

    def evaluate_genome(self, genome: Genome) -> float:
        nn = NeuralNetwork(genome)
        evaluation = self.evaluator.evaluate(nn)
        return evaluation

    def check_for_stagnation(self):
        for specie in self.species:
            specie.recalculate_max_fitness()
            if specie.max_fitness <= specie.prev_max_fitness:
                specie.no_improvement_age += 1
            else:
                specie.no_improvement_age = 0
            specie.has_best = self.champions[-1] in specie.genomes

        self.species = list(filter(lambda s: s.no_improvement_age < config.stagnation_age or s.has_best, self.species))

    def find_champion(self):
        self.champions.append(max(self.genomes, key=lambda genome: genome.fitness))

    def look_for_solution(self):
        nn = NeuralNetwork(self.champions[-1])
        if self.evaluator.solve(nn):
            self.solved_at = self.generation_index

    def reproduce_offspring(self):
        total_average = sum(specie.avg_adjusted_fitness for specie in self.species)
        for specie in self.species:
            specie.offspring_number = int(round(len(self.genomes) * specie.avg_adjusted_fitness / total_average))
        self.species = list(filter(lambda s: s.offspring_number > 0, self.species))
        if config.reset_innovations:
            innov_tracker.reset_innovations()

        new_genomes_global = []
        for specie in self.species:
            specie.genomes.sort(key=lambda ind: ind.fitness, reverse=True)
            keep = max(1, int(round(len(specie.genomes) * config.genome_survival_rate)))
            pool = specie.genomes[:keep]
            if config.elitism_enabled and len(specie.genomes) >= 1:
                specie.genomes = specie.genomes[:1]
                new_genomes_global += specie.genomes
            else:
                specie.genomes = []

            while len(specie.genomes) < specie.offspring_number:
                new_genomes = []
                if len(pool) == 1:
                    child = deepcopy(pool[0])
                    child.mutate()
                    new_genomes.append(child)
                else:
                    parent1 = deepcopy(tournament_selection(pool, min(len(pool), config.max_tournament_champions)))
                    parent2 = deepcopy(tournament_selection(pool, min(len(pool), config.max_tournament_champions)))
                    child = genome_crossover(parent1, parent2)
                    child.mutate()
                    new_genomes.append(child)
                specie.genomes += new_genomes
                new_genomes_global += new_genomes
        self.genomes = new_genomes_global

    def evolve(self):
        print(f"genomes: {len(self.genomes)}; species: {len(self.species)}")
        self.speciate()
        self.evaluate_all_fitness()
        self.find_champion()
        self.look_for_solution()
        self.check_for_stagnation()
        if config.allow_age_fitness_ajustment:
            for specie in self.species:
                specie.age_fitness_adjustment()
        self.reproduce_offspring()
        self.generation_index += 1
