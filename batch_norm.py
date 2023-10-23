"""
This module shows the various ways normalization could be implemented
in the previous MLP architecture.
# TO DO: add the remaining plots
# TO DO: do the youtube exercises
Exercises:
E01: I did not get around to seeing what happens when you initialize all weights and biases to zero.
Try this and train the neural net. You might think either that 1) the network trains just fine or 2) 
the network doesn't train at all, but actually it is 3) the network trains but only partially, and 
achieves a pretty bad final performance. Inspect the gradients and activations to figure out what is 
happening and why the network is only partially training, and what part is being trained exactly.
- E02: BatchNorm, unlike other normalization layers like LayerNorm/GroupNorm etc. has the big 
advantage that after training, the batchnorm gamma/beta can be "folded into" the weights of the 
preceeding Linear layers, effectively erasing the need to forward it at test time. Set up a small 
3-layer MLP with batchnorms, train the network, then "fold" the batchnorm gamma/beta into the 
preceeding Linear layer's W,b by creating a new W2, b2 and erasing the batch norm. Verify that this 
gives the same forward pass during inference. i.e. we see that the batchnorm is there just for 
stabilizing the training, and can be thrown out after training is done! pretty cool.
# TO DO: Read the mentioned ML papers(Delving deep into rectifiers, rethinking batch norm, layer 
norm group norm, instance norm)
"""

# pylint: disable=redefined-outer-name,too-many-instance-attributes, too-many-locals, too-many-arguments
import random
from typing import Tuple, List, Union
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {DEVICE}...")
SPECIAL_TOKEN = "<>"
BLOCK_SIZE = 3
HIDDEN_LAYER = 200
EMBEDDING_LENGTH = 10
LEARNING_RATE = 0.3669143319129944
EPOCHS = 1000
GENERATOR = torch.Generator(device=DEVICE).manual_seed(10)


class Embedding:
    """
    Class to define tanh layer
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.embedding = torch.randn(
            (num_embeddings, embedding_dim), device=DEVICE, generator=GENERATOR
        )
        self.output = None

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        self.output = self.embedding[inputs]
        self.output = self.output.view(-1, BLOCK_SIZE * EMBEDDING_LENGTH)
        return self.output

    def parameters(self) -> List[torch.Tensor]:
        """
        Function to return the parameters of the layer
        """
        return [self.embedding]


class Linear:
    """
    Class to define a linear layer
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.weight = torch.normal(
            0,
            (1 / (in_features**0.5)),
            (in_features, out_features),
            generator=GENERATOR,
            device=DEVICE,
        )
        self.bias = torch.zeros(out_features, device=DEVICE) if bias else None
        self.output = None

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        self.output = inputs @ self.weight
        if self.bias is not None:
            self.output += self.bias
        return self.output

    def parameters(self) -> List[torch.Tensor]:
        """
        Function to return the parameters of the layer
        """
        return [self.weight, self.bias] if self.bias is not None else [self.weight]


class BatchNorm:
    """
    Class to define batch norm layer
    """

    def __init__(
        self, num_features: int, eps: float = 1e-05, momentum: float = 0.1
    ) -> None:
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(num_features, device=DEVICE)
        self.beta = torch.zeros(num_features, device=DEVICE)
        self.running_mean = torch.zeros(num_features, device=DEVICE)
        self.running_var = torch.ones(num_features, device=DEVICE)
        self.output = None

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_mean = inputs.mean(dim=0, keepdim=True)
            batch_var = inputs.var(dim=0, keepdim=True)
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        norm_input = (inputs - batch_mean) / torch.sqrt(batch_var + self.eps)
        self.output = self.gamma * norm_input + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = ((1 - self.momentum) * self.running_mean) + (
                    self.momentum * batch_mean
                )
                self.running_var = ((1 - self.momentum) * self.running_var) + (
                    self.momentum * batch_var
                )
        return self.output

    def parameters(self) -> List[torch.Tensor]:
        """
        Function to return the parameters of the layer
        """
        return [self.gamma, self.beta]

    def eval(self) -> None:
        """
        Function to set layer to eval mode by changing self.training to false
        """
        self.training = False

    def train(self) -> None:
        """
        Function to set layer to train mode by changing self.training to true
        """
        self.training = True


class Tanh:
    """
    Class to define tanh layer
    """

    def __init__(self) -> None:
        self.output = None

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        self.output = torch.tanh(inputs)
        return self.output

    def parameters(self) -> List:
        """
        Function to return the parameters of the layer
        """
        return []


def split_data(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    proportions: Tuple[float] = (0.8, 0.1, 0.1),
    shuffle: bool = True,
) -> Tuple:
    """
    Function to split data into train, validation and test set.

    Args:
        features torch.Tensor: Inputs into the model.
        labels torch.Tensor: labels to the inputs.
        proportions (List[int], optional): Proportions for the split in this order,
        [train_set, validation_set, test_set]. Defaults to [0.8,0.1,0.1].

    Returns:
        Tuple: The splitted data in this order, ((train_features, train_labels),
        (validation_features, validation_labels), (test_features, test_labels))
    """
    assert len(inputs) == len(outputs), "The length of features and labels aren't equal"
    indexes = list(range(len(inputs)))
    if shuffle:
        random.shuffle(indexes)

    train_indexes = indexes[: int(len(indexes) * proportions[0])]
    validation_indexes = indexes[
        int(len(indexes) * proportions[0]) : int(
            len(indexes) * (proportions[0] + proportions[1])
        )
    ]
    test_indexes = indexes[int(len(indexes) * (proportions[0] + proportions[1])) :]

    train_features = features[train_indexes]
    train_labels = labels[train_indexes]

    validation_features = features[validation_indexes]
    validation_labels = labels[validation_indexes]

    test_features = features[test_indexes]
    test_labels = labels[test_indexes]

    return (
        (train_features, train_labels),
        (validation_features, validation_labels),
        (test_features, test_labels),
    )


def train(
    model: List[Union[Embedding, Linear, BatchNorm, Tanh]],
    training_set: Tuple[torch.Tensor],
    epochs: int,
    learning_rate: float,
    batch_size: int = 128,
) -> Tuple[Union[torch.Tensor, dict]]:
    """
    Function to train our classifier

    Args:
        model (List[Union[Embedding, Linear, BatchNorm, Tanh]]): Model layers as a
        list.
        training_set (Tuple[torch.Tensor]): dataset used to train classifier
        as a tuple of the features and labels i.e features, labels
        epochs (int): Number of iterations to train for
        learning_rate (float): Value controlling the rate of change of weights
        at each epoch
        batch_size (int): The size of training data to optimize on at a particular
        instance. It defaults to 128.
        analysis (bool): Denotes whether we are analysing the model or actually
        training it. It defaults to False.

    Returns:
       Tuple[Union[torch.Tensor, dict]]: A tuple of the models weights,
        and a dictionary containing both training and validations losses recorded during training.
    """
    losses = {"training loss": [], "validation loss": []}
    decayed_learning_rate = learning_rate * 0.1
    decayed_learning_rate_2 = learning_rate * 0.1**2
    parameters = [parameter for layer in model for parameter in layer.parameters()]
    print(
        f"\nTraining model with {sum(parameter.nelement() for parameter in parameters)} parameters"
    )
    for parameter in parameters:
        parameter.requires_grad = True
    for epoch in range(1, epochs + 1):
        batch_index = torch.randint(0, training_set[0].shape[0], (batch_size,))

        x, y = training_set[0][batch_index], training_set[1][batch_index]
        for layer in model:
            x = layer(x)

        train_loss = F.cross_entropy(x, y)
        losses["training loss"].append(train_loss.item())

        for layer in model:
            layer.output.retain_grad()
        for parameter in parameters:
            parameter.grad = None
        train_loss.backward()

        if 35000 < epoch < 70000:
            learning_rate = decayed_learning_rate
        elif epoch > 70000:
            learning_rate = decayed_learning_rate_2
        for parameter in parameters:
            parameter.data += -learning_rate * parameter.grad

        if epoch % 1000 == 0 or epoch == 1:
            print(
                f"Epoch {epoch} Training Loss: {train_loss}"
            )

    return model, losses


with open("names.txt", mode="r", encoding="utf-8") as file:
    data = file.read().splitlines()

unique_characters = [SPECIAL_TOKEN] + sorted(list(set("".join(data))))
index_to_character = dict(enumerate(unique_characters))
character_to_index = {
    character: index for index, character in index_to_character.items()
}

features = []
labels = []
for word in data:
    context = BLOCK_SIZE * [0]
    word = list(word) + ["<>"]
    for character in word:
        character_index = character_to_index[character]
        features.append(context)
        labels.append(character_index)
        context = context[1:] + [character_index]
features = torch.tensor(features, device=DEVICE)
labels = torch.tensor(labels, device=DEVICE)

model = [
    Embedding(len(unique_characters), EMBEDDING_LENGTH),
    Linear(EMBEDDING_LENGTH * BLOCK_SIZE, HIDDEN_LAYER),
    BatchNorm(HIDDEN_LAYER),
    Tanh(),
    Linear(HIDDEN_LAYER, HIDDEN_LAYER),
    BatchNorm(HIDDEN_LAYER),
    Tanh(),
    Linear(HIDDEN_LAYER, HIDDEN_LAYER),
    BatchNorm(HIDDEN_LAYER),
    Tanh(),
    Linear(HIDDEN_LAYER, HIDDEN_LAYER),
    BatchNorm(HIDDEN_LAYER),
    Tanh(),
    Linear(HIDDEN_LAYER, HIDDEN_LAYER),
    BatchNorm(HIDDEN_LAYER),
    Tanh(),
    Linear(HIDDEN_LAYER, len(unique_characters)),
    BatchNorm(len(unique_characters)),
]
with torch.no_grad():
    model[-1].gamma *= 0.1
    for layer in model[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 1.0


train_set, validation_set, test_set = split_data(features, labels)
model, _ = train(
    model=model,
    training_set=train_set,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
)

print("\nActivation distribution stats")
plt.figure(figsize=(20, 5))
legend = []
for index, layer in enumerate(model):
    if isinstance(layer, Tanh):
        tanh_output = layer.output
        print(
            f"layer {index} | Mean {tanh_output.mean():.2f} | "
            f"Standard deviation {tanh_output.std():.2f} | "
            f"Saturation {(tanh_output.abs() > 0.97).float().mean()}"
        )
        y, x = torch.histogram(tanh_output.cpu(), density=True)
        sns.lineplot(x=x[:-1].detach(), y=y.detach(), label = f"layer {index}")
        legend.append(f"layer {index}")
plt.title("Activation Distribution")
plt.savefig("images/activation_distributions")

print("\nActivation gradient distribution stats")
plt.figure(figsize=(20, 5))
legend = []
for index, layer in enumerate(model):
    if isinstance(layer, Tanh):
        tanh_output_grad = layer.output.grad
        print(
            f"layer {index} | Mean {tanh_output_grad.mean():.2f} | "
            f"Standard deviation {tanh_output_grad.std():.4f}"
        )
        y, x = torch.histogram(tanh_output_grad.cpu(), density=True)
        sns.lineplot(x=x[:-1].detach(), y=y.detach(), label = f"layer {index}")
        legend.append(f"layer {index}")
plt.title("Activation Gradient Distribution")
plt.savefig("images/activation_gradient_distributions")

with torch.no_grad():
    for layer in model:
        if isinstance(layer, BatchNorm):
            layer.eval()

    val_x, val_y = validation_set[0], validation_set[1]
    for layer in model:
        val_x = layer(val_x)

    val_loss = F.cross_entropy(val_x, val_y)
    print(f"\nValidation loss: {val_loss}")