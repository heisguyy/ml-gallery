"""
This is a code-along implemeentation of backpropagation algorithm for a neural network.
https://www.youtube.com/watch?v=VMj-3S1tku0
"""
# pylint: disable=protected-access
import math
from typing import Union, Tuple
from graphviz import Digraph

class Scaler:
    """
    This is a wrapper around a scaler value. It is a more toned down version
    of a matrix which is what you will expect in an actual neural network.
    """
    def __init__(
        self,
        data: Union[int,float],
        _children: Tuple["Scaler","Scaler"] = (),
        _op: str = "",
        label: str = "",
        grad: float = 0.0
    ):
        self.data = data
        self._children = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.label = label
        self.grad = grad

    def __repr__(self) -> str:
        return f"Scaler({self.label}): ({self.data})\n"

    def __add__(self,other) -> "Scaler":
        other = other if isinstance(other, Scaler) else Scaler(other)
        out = Scaler(self.data + other.data, (self,other), "+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self,other) -> "Scaler":
        return self + other

    def __neg__(self) -> "Scaler":
        return self * -1

    def __sub__(self,other) -> "Scaler":
        return self + (-other)

    def __mul__(self,other) -> "Scaler":
        other = other if isinstance(other, Scaler) else Scaler(other)
        out = Scaler(self.data * other.data, (self,other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self,other) -> "Scaler":
        return self * other

    def __truediv__(self,other) -> "Scaler":
        other = other if isinstance(other, Scaler) else Scaler(other)
        return self * other ** -1

    def __rtruediv__(self,other) -> "Scaler":
        return other * self ** -1

    def __pow__(self,other) -> "Scaler":
        assert isinstance(other, (int,float))
        out = Scaler(self.data ** other, (self,), f"**{other}")
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
        tan_h = (math.exp(2*self.data) - 1)/(math.exp(2*self.data) + 1)
        out = Scaler(tan_h, (self,), "tanh")
        def _backward():
            self.grad += (1 - (out.data ** 2)) * out.grad
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
                visited.insert(current_node)
                for child in current_node._children:
                    topological_sort(child)
                sorted_nodes.append(current_node)
        topological_sort(self)
        self.grad = 1.0
        for node in reversed(sorted_nodes):
            node._backward()

def trace(root: Scaler) -> Tuple[set,set]:
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

def draw_dot(root: Scaler, file_name: str = "result") -> Digraph:
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
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
    nodes, edges = trace(root)
    for node in nodes:
        uid = str(id(node))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name = uid,
            label = f" {node.label} | data: { node.data:.4f} | grad: {node.grad:.4f} ",
            shape = 'record'
        )
        if node._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + node._op, label = node._op)
            # and connect this node to it
            dot.edge(uid + node._op, uid)

    for node_1, node_2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(node_1)), str(id(node_2)) + node_2._op)

    dot.render(filename=file_name, format='png', cleanup=True)

if __name__ == "__main__":
    a = Scaler(2.0, label='a')
    bias = Scaler(-3.0, label='b')
    c = Scaler(10.0, label='c')
    e = a * bias
    e.label = 'e'
    d = e + c
    d.label = 'd'
    f = Scaler(-2.0, label='f')
    L = d * f
    L.label = 'L'
    # Manual Back propagation
    # dL/dL = 1.0
    L.grad = 1.0
    # dL/df ?
    # L = d * f
    # dL/df = d * 1.0
    # dL/df = d
    f.grad = d.data
    d.grad = f.data
    # dL/dc = dL/dd * dd/dc -> Chain Rule
    # d = e + c
    # dd/dc = 1.0
    # dL/dc = dL/dd * 1.0
    c.grad = d.grad * 1.0
    e.grad = d.grad * 1.0
    # dL/da = dL/de * de/da -> Chain Rule
    # e = a * b
    # de/da = b
    # dL/da = dL/de * b
    a.grad = e.grad * bias.data
    bias.grad = e.grad * a.data
    draw_dot(L, file_name="simple")
    # Single optimization step - Goal is to slightly increase the value of L
    a.data += 0.01 * a.grad
    bias.data += 0.01 * bias.grad
    c.data += 0.01 * c.grad
    f.data += 0.01 * f.grad
    e = a * bias
    d = e + c
    new_L = d * f
    print(f"Old L: {L.data}, New L: {new_L.data}")
    # More complex manual example - A perceptron
    # inputs to the perceptron
    input_1 = Scaler(2.0, label='x1')
    input_2 = Scaler(0.0, label='x2')
    # weights of the perceptron
    weight_1 = Scaler(-3.0, label='w1')
    weight_2 = Scaler(1.0, label='w2')
    # bias of the perceptron
    bias = Scaler(6.8813735870195432, label='b')
    # x1*w1 + x2*w2
    input_weight_1 = input_1 * weight_1
    input_weight_1.label = 'x1*w1'
    input_weight_2 = input_2 * weight_2
    input_weight_2.label = 'x2*w2'
    inputs_weights = input_weight_1 + input_weight_2
    inputs_weights.label = 'x1*w1 + x2*w2'
    # x1*w1 + x2*w2 + b
    output = inputs_weights + bias
    output.label = 'n'
    # tanh(x1*w1 + x2*w2 + b)
    activated_output = output.tanh()
    activated_output.label = 'o'
    # d(activated_output)/d(activated_output) = 1.0
    activated_output.grad = 1.0
    # activated_output = tanh(output)^2
    # d(activated_output)/d(output) = 1 - tanh(output)^2
    output.grad = 1 - (activated_output.data ** 2)
    # d(activated_output)/d(inputs_weight) = d(activated_output)/d(output) *
    #                                           d(output)/d(inputs_weight) -> Chain Rule
    # output = inputs_weights + bias
    # d(output)/d(inputs_weight) = 1.0
    inputs_weights.grad = output.grad * 1.0
    bias.grad = output.grad * 1.0
    # d(activated_output)/d(input_weight_1) = d(activated_output)/d(inputs_weights)
    #                                           * d(inputs_weights)/d(input_weight_1) -> Chain Rule
    # inputs_weights = input_weight_1 + input_weight_2
    # d(inputs_weights)/d(input_weight_1) = 1.0
    input_weight_1.grad = inputs_weights.grad * 1.0
    input_weight_2.grad = inputs_weights.grad * 1.0
    # d(activated_output)/d(input_1) = d(activated_output)/d(input_weight_1)
    #                                   * d(input_weight_1)/d(input_1) -> Chain Rule
    # input_weight_1 = input_1 * weight_1
    # d(input_weight_1)/d(input_1) = weight_1
    input_1.grad = input_weight_1.grad * weight_1.data
    weight_1.grad = input_weight_1.grad * input_1.data
    input_2.grad = input_weight_2.grad * weight_2.data
    weight_2.grad = input_weight_2.grad * input_2.data
    draw_dot(activated_output, file_name="perceptron")

    # Automated example - A perceptron
    # inputs to the perceptron
    input_1 = Scaler(2.0, label='x1')
    input_2 = Scaler(0.0, label='x2')
    # weights of the perceptron
    weight_1 = Scaler(-3.0, label='w1')
    weight_2 = Scaler(1.0, label='w2')
    # bias of the perceptron
    bias = Scaler(6.8813735870195432, label='b')
    # x1*w1 + x2*w2
    input_weight_1 = input_1 * weight_1
    input_weight_1.label = 'x1*w1'
    input_weight_2 = input_2 * weight_2
    input_weight_2.label = 'x2*w2'
    inputs_weights = input_weight_1 + input_weight_2
    inputs_weights.label = 'x1*w1 + x2*w2'
    # x1*w1 + x2*w2 + b
    output = inputs_weights + bias
    output.label = 'n'
    # tanh(x1*w1 + x2*w2 + b)
    activated_output = output.tanh()
    activated_output.label = 'o'

    activated_output.backward()
    draw_dot(activated_output, file_name="perceptron_auto")

    # Edge case on why we need to deposit gradients
    a = Scaler(2.0, label='a')
    b = a + a
    b.label = 'b'
    b.backward()
    draw_dot(b, file_name="edge_case")

    # Edge case on pure python values and Scalar objects
    a = Scaler(2.0, label='a')
    b = a + 2.0
    print(b)

    # Edge cases on the order of operations
    a = Scaler(2.0, label='a')
    b = 1 + a
    print(b)

    # Breaking down the tanh to its exponential components
    # inputs to the perceptron
    input_1 = Scaler(2.0, label='x1')
    input_2 = Scaler(0.0, label='x2')
    # weights of the perceptron
    weight_1 = Scaler(-3.0, label='w1')
    weight_2 = Scaler(1.0, label='w2')
    # bias of the perceptron
    bias = Scaler(6.8813735870195432, label='b')
    # x1*w1 + x2*w2
    input_weight_1 = input_1 * weight_1
    input_weight_1.label = 'x1*w1'
    input_weight_2 = input_2 * weight_2
    input_weight_2.label = 'x2*w2'
    inputs_weights = input_weight_1 + input_weight_2
    inputs_weights.label = 'x1*w1 + x2*w2'
    # x1*w1 + x2*w2 + b
    output = inputs_weights + bias
    output.label = 'n'
    # exp(2 * (x1*w1 + x2*w2 + b))
    double_output = output * 2
    double_output.label = '2n'
    exp_output = (double_output).exp()
    exp_output.label = 'exp_n'
    activated_output = (exp_output-1) / (exp_output + 1)
    activated_output.label = 'o'
    activated_output.backward()
    draw_dot(activated_output, file_name="perceptron_broken")
