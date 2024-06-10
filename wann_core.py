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
        )
        self.connections[connection.connection_id] = connection
        return connection