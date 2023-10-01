"""
An implementation of neural networks from scratch.
This is a code along with Andrej Karpathy's zero to hero neural network series.
https://www.youtube.com/watch?v=VMj-3S1tku0
"""

# pylint: disable=protected-access
import math
import random
from typing import List, Tuple, Union
from graphviz import Digraph


class Scaler:
    """
    This is a wrapper around a scaler value. It is a more toned down version
    of a matrix which is what you will expect in an actual neural network.
    """

    def __init__(
        self,
        data: Union[int, float],
        _children: Tuple["Scaler", "Scaler"] = (),
        _op: str = "",
        label: str = "",
        grad: float = 0.0,
    ):
        self.data = data
        self._children = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.label = label
        self.grad = grad

    def __repr__(self) -> str:
        return f"Scaler({self.label}): ({self.data})\n"

    def __add__(self, other) -> "Scaler":
        other = other if isinstance(other, Scaler) else Scaler(other)
        out = Scaler(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other) -> "Scaler":
        return self + other

    def __neg__(self) -> "Scaler":
        return self * -1

    def __sub__(self, other) -> "Scaler":
        return self + (-other)

    def __mul__(self, other) -> "Scaler":
        other = other if isinstance(other, Scaler) else Scaler(other)
        out = Scaler(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other) -> "Scaler":
        return self * other

    def __truediv__(self, other) -> "Scaler":
        other = other if isinstance(other, Scaler) else Scaler(other)
        return self * other**-1

    def __rtruediv__(self, other) -> "Scaler":
        return other * self**-1

    def __pow__(self, other) -> "Scaler":
        assert isinstance(other, (int, float))
        out = Scaler(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Scaler":
        """
        Function that implements the forward and backward pass of tanh activation function.

        Returns:
            Scaler: A Scaler object with the data being the result of the tanh function.
        """
        tan_h = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Scaler(tan_h, (self,), "tanh")

        def _backward():
            self.grad += (1 - (out.data**2)) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Scaler":
        """
        Function that implements the forward and backward pass of exponential function.

        Returns:
            Scaler: A Scaler object with the data being the result of the exponential function.
        """
        out = Scaler(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """
        Function to perform the backward pass of the neural network.
        """
        sorted_nodes = []
        visited = []

        def topological_sort(current_node):
            if current_node not in visited:
                visited.append(current_node)
                for child in current_node._children:
                    topological_sort(child)
                sorted_nodes.append(current_node)

        topological_sort(self)
        self.grad = 1.0
        for node in reversed(sorted_nodes):
            node._backward()


def trace(root: Scaler) -> Tuple[set, set]:
    """
    Function to recursively trace the graph of a neural network.

    Args:
        root (Scalar): Scalar object that is the root of the graph.

    Returns:
        Tuple[set,set]: A tuple where the first element is a set of all the nodes
        and the second element is the set of all edges.
    """
    nodes, edges = set(), set()

    def build(current_node):
        if current_node not in nodes:
            nodes.add(current_node)
            for child in current_node._children:
                edges.add((child, current_node))
                build(child)

    build(root)
    return nodes, edges


def draw_graph(root: Scaler, file_name: str = "result") -> Digraph:
    """
    Function to draw the graph of a neural network.

    Args:
        root (Scalar): Scalar object that is the root of the graph.
        file_name (str, optional): File name to save results too.
        Defaults to "result.svg".

    Returns:
        Digraph: A graphviz Digraph object which shows the relation in the neural network
        by connecting the nodes with edges.
    """
    graph = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right
    nodes, edges = trace(root)
    for node in nodes:
        uid = str(id(node))
        # for any value in the graph, create a rectangular ('record') node for it
        graph.node(
            name=uid,
            label=f" {node.label} | data: { node.data:.4f} | grad: {node.grad:.4f} ",
            shape="record",
        )
        if node._op:
            # if this value is a result of some operation, create an op node for it
            graph.node(name=uid + node._op, label=node._op)
            # and connect this node to it
            graph.edge(uid + node._op, uid)

    for node_1, node_2 in edges:
        # connect n1 to the op node of n2
        graph.edge(str(id(node_1)), str(id(node_2)) + node_2._op)

    graph.render(filename=file_name, format="png", cleanup=True)


class Neuron:
    """
    This class is an implementation of a neuron in a neural network.
    """

    def __init__(self, number_of_inputs: int) -> None:
        self.weights = [Scaler(random.uniform(-1, 1)) for _ in range(number_of_inputs)]
        self.bias = Scaler(random.uniform(-1, 1))

    def __call__(self, inputs: List) -> Scaler:
        output = sum([w * x for w, x in zip(self.weights, inputs)], self.bias)
        activated_output = output.tanh()
        return activated_output

    def parameters(self) -> List[Scaler]:
        """
        Function to return all the parameters(weights and bias) for the neuron.

        Returns:
            List[Scaler]: A list of the parameters
        """
        neuron_paramters = self.weights + [self.bias]
        return neuron_paramters


class Layer:
    """
    This class is an implementation of a layer in a neural network.
    """

    def __init__(self, number_of_inputs: int, number_of_neurons: int) -> None:
        self.neurons = [Neuron(number_of_inputs) for _ in range(number_of_neurons)]

    def __call__(self, inputs: List) -> Union[List, Scaler]:
        out = [neuron(inputs) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> List[Scaler]:
        """
        Function to return all the neuron parameters.

        Returns:
            List[Scaler]: This is a list of all the parameters in a layer.
        """
        layer_parameters = [
            parameter for neuron in self.neurons for parameter in neuron.parameters()
        ]
        return layer_parameters


class MLP:
    """
    This class is an implementation of a multi-layer perceptron.
    """

    def __init__(
        self, number_of_inputs: int, hidden_layers: List, number_of_outputs: int
    ) -> None:
        layers_definition = [number_of_inputs] + hidden_layers + [number_of_outputs]
        self.layers = [
            Layer(layers_definition[i], layers_definition[i + 1])
            for i in range(len(layers_definition) - 1)
        ]

    def __call__(self, inputs: List) -> Union[List, Scaler]:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self) -> List[Scaler]:
        """
        Function to return all the parameters in a layer.

        Returns:
            List[Scaler]: A list of all the parameters in a network
        """
        network_parameters = [
            parameter for layer in self.layers for parameter in layer.parameters()
        ]
        return network_parameters


def train(model: MLP, data: List[List], targets: List, epochs: int = 10):
    """
    Function to train the MLP using simple gradient descent.

    Args:
        model (MLP): The model to train
        data (List[List]): List of features
        targets (List): The target for those features
        epochs (int, optional): Number of times to go over the forward and backward pass.
        Defaults to 10.
    """
    learning_rate = 0.1
    for epoch in range(1, epochs + 1):
        y_preds = [model(feature) for feature in data]
        loss = sum((y_pred - target) ** 2 for y_pred, target in zip(y_preds, targets))
        print(f"Epoch {epoch}: {loss.data}")
        for parameter in model.parameters():
            parameter.grad = 0.0
        loss.backward()
        for parameter in model.parameters():
            parameter.data += -learning_rate * parameter.grad
