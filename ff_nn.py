import pickle
from config import config
from wann_core import Genome
import matplotlib.pyplot as plt
import networkx as nx


class NeuralNetwork:
    def __init__(self, genome: Genome):
        self.genome = genome
        self.nodes = {node.node_id: node for node in genome.nodes}
        self.connections = [
            conn for conn in genome.connections.values() if conn.enabled
        ]

    def set_all_weights(self, weight: float):
        for connection in self.connections:
            connection.weight = weight

    def feed(self, inputs: list[float]) -> list[float]:
        node_values = {node.node_id: 0.0 for node in self.nodes.values()}

        input_nodes = [node for node in self.nodes.values() if node.layer == 0]
        for i, input_value in enumerate(inputs):
            node_values[input_nodes[i].node_id] = input_value

        if config.add_bias_node:
            node_values[input_nodes[-1].node_id] = config.bias_value

        layers = sorted(set(node.layer for node in self.nodes.values()))
        for layer in layers[1:]:  # Skip input layer
            for node in [n for n in self.nodes.values() if n.layer == layer]:
                input_sum = sum(
                    node_values[conn.from_node.node_id] * conn.weight
                    for conn in self.connections
                    if conn.to_node.node_id == node.node_id
                )
                node_values[node.node_id] = node.activation_f.value(
                    input_sum * node.activation_response
                )

        output_nodes = [
            node for node in self.nodes.values() if node.layer == max(layers)
        ]

        return [node_values[node.node_id] for node in output_nodes]
    
    def visualize(self, show_weights=True):
        G = nx.DiGraph()

        pos = {}
        labels = {}

        layers = sorted(set(node.layer for node in self.nodes.values()))
        layer_nodes = {
            layer: [node for node in self.nodes.values() if node.layer == layer]
            for layer in layers
        }

        max_nodes_in_layer = max(len(nodes) for nodes in layer_nodes.values())
        horizontal_spacing = 2
        vertical_spacing = 2

        for layer in layers:
            nodes = layer_nodes[layer]
            num_nodes = len(nodes)
            y_offset = (max_nodes_in_layer - num_nodes) * vertical_spacing / 2
            for i, node in enumerate(nodes):
                pos[node.node_id] = (
                    layer * horizontal_spacing,
                    i * vertical_spacing + y_offset,
                )
                if layer == 0 and i == len(nodes) - 1 and config.add_bias_node:
                    labels[node.node_id] = "bias"
                elif layer == 0:
                    labels[node.node_id] = f"{node.node_id} ({node.layer})"
                else:
                    labels[node.node_id] = f"{node.node_id} ({node.layer})\n{node.activation_f.name}"

        edge_labels = {}
        for conn in self.connections:
            edge_color = "red" if conn.weight > 0 else "black"
            G.add_edge(conn.from_node.node_id, conn.to_node.node_id, color=edge_color)
            if show_weights:
                edge_labels[(conn.from_node.node_id, conn.to_node.node_id)] = (
                    f"{conn.weight:.2f}"
                )

        edges = G.edges()
        colors = [G[u][v]["color"] for u, v in edges]

        plt.figure(figsize=(12, 8))
        nx.draw(
            G,
            pos,
            with_labels=True,
            labels=labels,
            node_size=3000,
            node_color="lightblue",
            font_size=10,
            font_weight="bold",
            arrowsize=20,
            edge_color=colors,
        )
        if show_weights:
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, font_color="blue"
            )

        plt.title("Neural Network Visualization")
        plt.show()
    
    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.genome, f)

    @staticmethod
    def load(filename: str) -> "NeuralNetwork":
        with open(filename, 'rb') as f:
            genome = pickle.load(f)
        return NeuralNetwork(genome)