from dataclasses import dataclass


@dataclass
class Configuration:
    input_nodes: int = 8
    output_nodes: int = 4
    add_bias_node: bool = True
    bias_value: float = 1
    max_depth: int = 10
    weight_range: tuple[float, float] = (-2, 2)
    initial_connection_prob: float = 1
    max_activation_response_delta: float = 0.3
    activation_response_range: tuple[float, float] = (0, 1)
    prob_reenable_connection: float = 0.15
    population_size: int = 100
    distance_excess: float = 1.0
    distance_disjoint: float = 1.0
    distance_activation: float = 0.4
    compatibility_threshold: float = 20
    compatibility_threshold_delta: float = 3
    min_compatibility_threshold: float = 0.1
    mutation_add_connection_prob: float = 0.25
    mutation_split_connection_prob: float = 0.2
    mutation_disable_connection_prob: float = 0.20
    mutation_enable_connection_prob: float = 0.20
    mutation_change_activation_f_prob: float = 0.35
    mutation_change_activation_response_prob: float = 0.3
    single_structure_mutation: bool = False
    reset_innovations: bool = True
    target_species: int = 15
    elitism_enabled: bool = True
    genome_survival_rate: float = 0.3
    allow_age_fitness_ajustment: bool = True
    young_age: int = 5
    young_multiplier: float = 1.2
    old_age: int = 15
    old_multiplier: float = 0.5
    max_tournament_champions: int = 3

    stagnation_age: int = 15

    wann_step: float = 0.5
    wann_random_activation_on_init: bool = True
    wann_initial_weight = 1
    wann_weights_pool = [-2, -1, 1, 2]
    wann_get_node_from_more_fit = False
    wann_use_custom_fitness = False


config = Configuration()
