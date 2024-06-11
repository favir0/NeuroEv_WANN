import random 

from activation import ActivationF


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

    def add_node(self, layer: int) -> Node:
        node = Node(node_id=len(self.nodes), layer=layer)
        node.activation_f = ActivationF.TANH
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