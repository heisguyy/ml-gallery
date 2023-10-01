"""
This is a code-along manual implemeentation of backpropagation algorithm for a neural network.
https://www.youtube.com/watch?v=VMj-3S1tku0
"""
from neural_network import MLP, Layer, Neuron, Scaler, draw_graph, train

a = Scaler(2.0, label="a")
bias = Scaler(-3.0, label="b")
c = Scaler(10.0, label="c")
e = a * bias
e.label = "e"
d = e + c
d.label = "d"
f = Scaler(-2.0, label="f")
L = d * f
L.label = "L"
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
draw_graph(L, file_name="images/simple")
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
input_1 = Scaler(2.0, label="x1")
input_2 = Scaler(0.0, label="x2")
# weights of the perceptron
weight_1 = Scaler(-3.0, label="w1")
weight_2 = Scaler(1.0, label="w2")
# bias of the perceptron
bias = Scaler(6.8813735870195432, label="b")
# x1*w1 + x2*w2
input_weight_1 = input_1 * weight_1
input_weight_1.label = "x1*w1"
input_weight_2 = input_2 * weight_2
input_weight_2.label = "x2*w2"
inputs_weights = input_weight_1 + input_weight_2
inputs_weights.label = "x1*w1 + x2*w2"
# x1*w1 + x2*w2 + b
output = inputs_weights + bias
output.label = "n"
# tanh(x1*w1 + x2*w2 + b)
activated_output = output.tanh()
activated_output.label = "o"
# d(activated_output)/d(activated_output) = 1.0
activated_output.grad = 1.0
# activated_output = tanh(output)^2
# d(activated_output)/d(output) = 1 - tanh(output)^2
output.grad = 1 - (activated_output.data**2)
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
draw_graph(activated_output, file_name="images/perceptron")

# Automated example - A perceptron
# inputs to the perceptron
input_1 = Scaler(2.0, label="x1")
input_2 = Scaler(0.0, label="x2")
# weights of the perceptron
weight_1 = Scaler(-3.0, label="w1")
weight_2 = Scaler(1.0, label="w2")
# bias of the perceptron
bias = Scaler(6.8813735870195432, label="b")
# x1*w1 + x2*w2
input_weight_1 = input_1 * weight_1
input_weight_1.label = "x1*w1"
input_weight_2 = input_2 * weight_2
input_weight_2.label = "x2*w2"
inputs_weights = input_weight_1 + input_weight_2
inputs_weights.label = "x1*w1 + x2*w2"
# x1*w1 + x2*w2 + b
output = inputs_weights + bias
output.label = "n"
# tanh(x1*w1 + x2*w2 + b)
activated_output = output.tanh()
activated_output.label = "o"

activated_output.backward()
draw_graph(activated_output, file_name="images/perceptron_auto")

# Edge case on why we need to deposit gradients
a = Scaler(2.0, label="a")
b = a + a
b.label = "b"
b.backward()
draw_graph(b, file_name="images/edge_case")

# Edge case on pure python values and Scalar objects
a = Scaler(2.0, label="a")
b = a + 2.0
print(b)

# Edge cases on the order of operations
a = Scaler(2.0, label="a")
b = 1 + a
print(b)

# Breaking down the tanh to its exponential components
# inputs to the perceptron
input_1 = Scaler(2.0, label="x1")
input_2 = Scaler(0.0, label="x2")
# weights of the perceptron
weight_1 = Scaler(-3.0, label="w1")
weight_2 = Scaler(1.0, label="w2")
# bias of the perceptron
bias = Scaler(6.8813735870195432, label="b")
# x1*w1 + x2*w2
input_weight_1 = input_1 * weight_1
input_weight_1.label = "x1*w1"
input_weight_2 = input_2 * weight_2
input_weight_2.label = "x2*w2"
inputs_weights = input_weight_1 + input_weight_2
inputs_weights.label = "x1*w1 + x2*w2"
# x1*w1 + x2*w2 + b
output = inputs_weights + bias
output.label = "n"
# exp(2 * (x1*w1 + x2*w2 + b))
double_output = output * 2
double_output.label = "2n"
exp_output = (double_output).exp()
exp_output.label = "exp_n"
activated_output = (exp_output - 1) / (exp_output + 1)
activated_output.label = "o"
activated_output.backward()
draw_graph(activated_output, file_name="images/perceptron_broken")

# Using the Neuron Class
inputs = [2.0, 0]
neuron = Neuron(2)
result = neuron(inputs)
draw_graph(result, "images/neuron")

# Using the Layer Class
inputs = [2.0, 0]
layer = Layer(2, 4)
result = layer(inputs)
print(result)

# Using the MLP class
inputs = [2.0, 0]
neural_network = MLP(2, [4, 4], 1)
result = neural_network(inputs=inputs)
print(result)
draw_graph(result, "images/MLP")

# Using MLP class, a little more complex example
neural_network = MLP(3, [4, 4], 1)
features = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
target = [1.0, -1.0, -1.0, 1.0]
train(neural_network, features, target)
predictions = [neural_network(feature) for feature in features]
print(f"\nPredictions: {predictions}")
