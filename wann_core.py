import random
import numpy as np

from copy import deepcopy
from activation import ActivationF
from configuration import config


class Node:
    def __init__(self, node_id: int, layer: int, activation_f: ActivationF = ActivationF.RELU, activation_response: float = 1):
        self.node_id = node_id
        self.layer = layer
        self.activation_f = activation_f
        self.activation_response = activation_response

class Connection:
    def __init__(self, from_node: Node, to_node: Node, weight: float, enabled: bool, innovation_number: int):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = innovation_number

    @property
    def connection_id(self):
        return self.from_node.node_id, self.to_node.node_id


class InnovTracker:
    def __init__(self):
        self.current_innovation_number = 0
        self.innovations = {}

    def get_innovation_number(self, in_node: Node, out_node: Node):
        key = (in_node.node_id, out_node.node_id)
        if key not in self.innovations:
            self.innovations[key] = self.current_innovation_number
            self.current_innovation_number += 1
        return self.innovations[key]

    def reset_innovations(self):
        self.innovations = {}
        self.current_innovation_number = 0

innov_tracker = InnovTracker()


class Genome:
    def __init__(self):
        self.connections: dict[tuple[int, int], Connection] = dict()
        self.nodes: list[Node] = list()
        self.fitness: float = 0
        self.adjusted_fitness: float = 0

    def add_node(self, layer: int) -> Node:
        node = Node(node_id=len(self.nodes), layer=layer)
        if (config.wann_random_activation_on_init):
            node.activation_f = random.choice(list(ActivationF))
        self.nodes.append(node)
        return node

    def add_connection(self, in_node: Node, out_node: Node, weight: float) -> Connection:
        connection = Connection(
            from_node=in_node,
            to_node=out_node,
            weight=weight,
            enabled=True,
            innovation_number=innov_tracker.get_innovation_number(
                in_node, out_node
            ),
        )
        self.connections[connection.connection_id] = connection
        return connection

    def mutation_add_connection(self):
        potential_connections = []
        for node_i in self.nodes:
            for node_j in self.nodes:
                if (
                    node_i.layer < node_j.layer
                    and (node_i.node_id, node_j.node_id)
                    not in self.connections
                ):
                    potential_connections.append((node_i, node_j))

        if not potential_connections:
            return

        in_node, out_node = random.choice(potential_connections)
        weight = config.wann_initial_weight
        self.add_connection(in_node, out_node, weight)

    def mutation_split_connection(self):
        possible_connections = []
        for connection in self.connections.values():
            new_layer = (connection.from_node.layer + connection.to_node.layer) // 2
            if new_layer != connection.from_node.layer:
                possible_connections.append(connection)

        if not possible_connections:
            return

        connection = random.choice(possible_connections)
        node = self.add_node(layer=connection.from_node.layer + 1)
        self.add_connection(in_node=connection.from_node, out_node=node, weight=config.wann_initial_weight)
        self.add_connection(in_node=node, out_node=connection.to_node, weight=connection.weight)

    def mutation_disable_connection(self):
        possible_connections = list(filter(lambda x: x.enabled, self.connections.values()))

        if not possible_connections:
            return

        connection = random.choice(list(possible_connections))
        connection.enabled = False

    def mutation_enable_connection(self):
        possible_connections = list(filter(lambda x: not x.enabled, self.connections.values()))

        if not possible_connections:
            return

        connection = random.choice(list(possible_connections))
        connection.enabled = True

    def mutation_change_activation_f(self):
        node = random.choice(self.nodes)
        activation_f = random.choice(list(ActivationF))
        node.activation_f = activation_f

    def mutation_change_activation_response(self):
        node = random.choice(self.nodes)
        response = node.activation_response + random.uniform(
            -config.max_activation_response_delta,
            config.max_activation_response_delta,
        )
        node.activation_response = np.clip(response, *config.activation_response_range)

    def mutate(self):
        if random.random() < config.mutation_disable_connection_prob:
            self.mutation_disable_connection()
        if random.random() < config.mutation_enable_connection_prob:
            self.mutation_enable_connection()
        if random.random() < config.mutation_change_activation_f_prob:
            self.mutation_change_activation_f()
        if random.random() < config.mutation_change_activation_response_prob:
            self.mutation_change_activation_response()

        if random.random() < config.mutation_add_connection_prob:
            self.mutation_add_connection()
            if config.single_structure_mutation:
                return
        if random.random() < config.mutation_split_connection_prob:
            self.mutation_split_connection()
            if config.single_structure_mutation:
                return

    def print_genome(self):
        print("Nodes:")
        for node in self.nodes:
            print(
                f'''Node ID: {node.node_id}, 
                Layer: {node.layer}, 
                Activation Function: {node.activation_f}, 
                Activation Response: {node.activation_response}''')

        print("\nConnections:")
        for conn_id, connection in self.connections.items():
            print(
                f'''Connection ID: {conn_id}, 
                From Node: {connection.from_node.node_id}, 
                To Node: {connection.to_node.node_id}, 
                Weight: {connection.weight}, 
                Enabled: {connection.enabled}, 
                Innovation Number: {connection.innovation_number}''')


def genome_crossover(genome0: Genome, genome1: Genome) -> Genome:
    child = Genome()

    if genome0.fitness > genome1.fitness:
        more_fit_parent = genome0
        less_fit_parent = genome1
    elif genome0.fitness < genome1.fitness:
        more_fit_parent = genome0
        less_fit_parent = genome1
        # if fitness is equal - let the more_fit be the larges one
    else:
        more_fit_parent = max(genome0, genome1, key=lambda x: len(x.nodes))
        less_fit_parent = min(genome0, genome1, key=lambda x: len(x.nodes))

    # ------- Adding nodes -------
    # Homologous gene: combine genes from both parents.
    for node0, node1 in zip(more_fit_parent.nodes, less_fit_parent.nodes):
        node = random.choice((node0, node1))
        child.nodes.append(deepcopy(node))

    # Excess or disjoint gene: copy from the fittest parent.
    # If fitnesses are equal - we can still copy more_fit because it always the largest one
    # and nodes are always sequential like 0 -> 1 -> 2 -> 3 etc
    if len(more_fit_parent.nodes) > len(less_fit_parent.nodes):
        for node in more_fit_parent.nodes[len(less_fit_parent.nodes) :]:
            child.nodes.append(deepcopy(node))

    # ------- Adding connections -------
    intersection = (more_fit_parent.connections.keys() & less_fit_parent.connections.keys())

    # Homologous gene: combine genes from both parents.
    for connection_id in intersection:
        connection = connection_crossover(connection_id, more_fit_parent, less_fit_parent)
        child.connections[connection_id] = connection

    # Excess or disjoint gene: copy from the fittest parent.
    # If fitnesses are equal - copy from both
    if more_fit_parent.fitness == less_fit_parent.fitness:
        connection_union = more_fit_parent.connections | less_fit_parent.connections
    # else only from more_fit
    else:
        connection_union = more_fit_parent.connections

    for connection_id, connection in connection_union.items():
        if connection_id not in intersection:
            child.connections[connection_id] = deepcopy(connection)

    return child

def connection_crossover(connection_id: tuple[int, int], genome0: Genome, genome1: Genome):
    connection = deepcopy(genome0.connections[connection_id])
    connection.enabled = (
        genome0.connections[connection_id].enabled
        and genome0.connections[connection_id].enabled
    ) or random.random() > config.prob_reenable_connection

    # don't care about weight for WANN
    return connection


# Stanley formula: delta = с1 * E / N + с2 * D / n + сk3 * W
# Where E - excess amount; D - disjoint amount; W - avgWeightDiff; N - matches; с1,с2,с3 - constant coefficients
def genome_distance(genome0: Genome, genome1: Genome):
    genome0_innovations = {c.innovation_number: c for c in genome0.connections.values()}
    genome1_innovations = {c.innovation_number: c for c in genome1.connections.values()}

    all_innovations = genome0_innovations | genome1_innovations

    min_innovation = min(max(genome0_innovations.keys()), max(genome1_innovations.keys()))

    excess = 0
    disjoint = 0
    avg_weight_diff = 0.0
    matches = 0

    for i in all_innovations.keys():
        if i in genome0_innovations and i in genome1_innovations:
            avg_weight_diff += np.abs(genome0_innovations[i].weight - genome1_innovations[i].weight)
            matches += 1
        else:
            if i <= min_innovation:
                disjoint += 1
            else:
                excess += 1

    avg_weight_diff = (avg_weight_diff / matches) if matches > 0 else avg_weight_diff

    return (
        config.distance_excess * excess
        + config.distance_disjoint * disjoint
        + config.distance_weight * avg_weight_diff
    )
