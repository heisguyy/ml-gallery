"""
This is an implementation of A Neural Probabilistic Language Model by Bengio et al, 2003.
It is a code along with Andrej Karpathy's zero to hero neural network series.
https://www.youtube.com/watch?v=TCH_1BHY58I&t=236s
"""
# pylint: disable=redefined-outer-name, invalid-name
import random
from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import wandb


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
    parameters: List[torch.Tensor],
    training_set: Tuple[torch.Tensor],
    val_set: Tuple[torch.Tensor],
    epochs: int,
    learning_rate: float,
    batch_size: int = 128,
) -> Tuple[torch.Tensor, dict]:
    """
    Function to train our classifier

    Args:
        parameters (List[torch.Tensor]): Model parameters including embedding, weights
        and biases. They should be in the sequence [embedding_layer, weights_1,
        bias_1, weights_2, bias_2]
        training_set (Tuple[torch.Tensor]): dataset used to train classifier
        as a tuple of the features and labels i.e features, labels
        val_set (Tuple[torch.Tensor]): dataset used to validate classifier
        as a tuple of the features and labels i.e features, labels
        epochs (int): Number of iterations to train for
        learning_rate (float): Value controlling the rate of change of weights
        at each epoch
        batch_size (int): The size of training data to optimize on at a particular
        instance. It defaults to 128.

    Returns:
        Tuple[torch.Tensor, dict]: A tuple of the models weights and a dictionary
        containing both training and validations losses recorded during training.
    """
    losses = {"training loss": [], "validation loss": []}
    decayed_learning_rate = learning_rate * 0.1
    decayed_learning_rate_2 = learning_rate * 0.1**2
    for epoch in range(1, epochs + 1):
        batch_index = torch.randint(0, training_set[0].shape[0], (batch_size,))
        embeddings = parameters[0][training_set[0][batch_index]]
        hidden_layer_output = torch.tanh(
            embeddings.view(-1, BLOCK_SIZE * EMBEDDING_LENGTH) @ parameters[1]
            + parameters[2]
        )
        logits = (
            embeddings.view(-1, BLOCK_SIZE * EMBEDDING_LENGTH) @ parameters[3]
            + hidden_layer_output @ parameters[4]
            + parameters[5]
        )
        train_loss = F.cross_entropy(logits, training_set[1][batch_index])
        losses["training loss"].append(train_loss.item())
        for parameter in parameters:
            parameter.grad = None
        train_loss.backward()
        if epoch > 100000 and epoch < 200000:
            learning_rate = decayed_learning_rate
        elif epoch > 200000:
            learning_rate = decayed_learning_rate_2

        for parameter in parameters:
            parameter.data += -learning_rate * parameter.grad
        with torch.no_grad():
            batch_index = torch.randint(0, val_set[0].shape[0], (batch_size,))
            embeddings = parameters[0][val_set[0][batch_index]]
            hidden_layer_output = torch.tanh(
                embeddings.view(-1, BLOCK_SIZE * EMBEDDING_LENGTH) @ parameters[1]
                + parameters[2]
            )
            logits = (
                embeddings.view(-1, BLOCK_SIZE * EMBEDDING_LENGTH) @ parameters[3]
                + hidden_layer_output @ parameters[4]
                + parameters[5]
            )
            val_loss = F.cross_entropy(logits, val_set[1][batch_index])
            losses["validation loss"].append(val_loss.item())
        if epoch % 1000 == 0 or epoch == 1:
            print(
                f"Epoch {epoch} Training Loss: {train_loss}, Validation Loss: {val_loss}"
            )

    return parameters, losses


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {DEVICE}...\n")

with open("names.txt", mode="r", encoding="utf-8") as file:
    data = file.read().splitlines()

SPECIAL_TOKEN = "<>"
unique_characters = [SPECIAL_TOKEN] + sorted(list(set("".join(data))))
index_to_character = dict(enumerate(unique_characters))
character_to_index = {
    character: index for index, character in index_to_character.items()
}

BLOCK_SIZE = 3
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

HIDDEN_LAYER = 200
EMBEDDING_LENGTH = 10
generator = torch.Generator(device=DEVICE).manual_seed(10)
embedding_layer = torch.normal(
    0,
    0.1,
    (len(unique_characters), EMBEDDING_LENGTH),
    device=DEVICE,
    requires_grad=True,
    generator=generator,
)
weights_1 = torch.normal(
    0,
    0.1,
    (BLOCK_SIZE * EMBEDDING_LENGTH, HIDDEN_LAYER),
    device=DEVICE,
    requires_grad=True,
    generator=generator,
)
bias_1 = torch.normal(
    0, 0.1, (1, HIDDEN_LAYER), device=DEVICE, requires_grad=True, generator=generator
)
weights_2 = torch.normal(
    0,
    0.1,
    (HIDDEN_LAYER, len(unique_characters)),
    device=DEVICE,
    requires_grad=True,
    generator=generator,
)
bias_2 = torch.normal(
    0,
    0.1,
    (1, len(unique_characters)),
    device=DEVICE,
    requires_grad=True,
    generator=generator,
)
direct_connection_weights = torch.normal(
    0,
    0.1,
    (BLOCK_SIZE * EMBEDDING_LENGTH, len(unique_characters)),
    device=DEVICE,
    requires_grad=True,
    generator=generator,
)
parameters = [
    embedding_layer,
    weights_1,
    bias_1,
    direct_connection_weights,
    weights_2,
    bias_2,
]

EPOCHS = 1000
learning_rate_exp = torch.linspace(-3, 0, EPOCHS, device=DEVICE)
learning_rates = 10**learning_rate_exp
losses = []
for epoch in range(1, EPOCHS + 1):
    batch_index = torch.randint(0, features.shape[0], (128,))
    embeddings = embedding_layer[features[batch_index]]
    hidden_layer_output = torch.tanh(
        embeddings.view(-1, BLOCK_SIZE * EMBEDDING_LENGTH) @ weights_1 + bias_1
    )
    logits = (
        embeddings.view(-1, BLOCK_SIZE * EMBEDDING_LENGTH) @ direct_connection_weights
        + hidden_layer_output @ weights_2
        + bias_2
    )
    loss = F.cross_entropy(logits, labels[batch_index])
    losses.append(loss.item())
    for parameter in parameters:
        parameter.grad = None
    loss.backward()
    learning_rate = learning_rates[epoch - 1]
    for parameter in parameters:
        parameter.data += -learning_rate * parameter.grad

plt.figure(figsize=(10, 10))
sns.lineplot(x=learning_rate_exp.cpu(), y=losses)
plt.savefig("images/learning_rate_search")

best_lr_exp = learning_rate_exp[losses.index(min(losses))]
LEARNING_RATE = 10**best_lr_exp
print(f"\nBest learning rate is {LEARNING_RATE}\n")

generator = torch.Generator(device=DEVICE).manual_seed(10)
embedding_layer = torch.normal(
    0,
    0.1,
    (len(unique_characters), EMBEDDING_LENGTH),
    device=DEVICE,
    requires_grad=True,
    generator=generator,
)
weights_1 = torch.normal(
    0,
    0.1,
    (BLOCK_SIZE * EMBEDDING_LENGTH, HIDDEN_LAYER),
    device=DEVICE,
    requires_grad=True,
    generator=generator,
)
bias_1 = torch.normal(
    0, 0.1, (1, HIDDEN_LAYER), device=DEVICE, requires_grad=True, generator=generator
)
weights_2 = torch.normal(
    0,
    0.1,
    (HIDDEN_LAYER, len(unique_characters)),
    device=DEVICE,
    requires_grad=True,
    generator=generator,
)
bias_2 = torch.normal(
    0,
    0.1,
    (1, len(unique_characters)),
    device=DEVICE,
    requires_grad=True,
    generator=generator,
)
direct_connection_weights = torch.normal(
    0,
    0.1,
    (BLOCK_SIZE * EMBEDDING_LENGTH, len(unique_characters)),
    device=DEVICE,
    requires_grad=True,
    generator=generator,
)
parameters = [
    embedding_layer,
    weights_1,
    bias_1,
    direct_connection_weights,
    weights_2,
    bias_2,
]
EPOCHS = 200000

wandb.init(
    project="ml-gallery",
    name="direct-conn",
    config={
        "BLOCK_SIZE": BLOCK_SIZE,
        "HIDDEN_LAYER": HIDDEN_LAYER,
        "EMBEDDING_LENGTH": EMBEDDING_LENGTH,
        "LEARNING_RATE": LEARNING_RATE,
        "EPOCHS": EPOCHS,
    },
)

train_set, validation_set, test_set = split_data(features, labels)
parameters, losses = train(
    parameters=parameters,
    training_set=train_set,
    val_set=validation_set,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
)

plt.figure(figsize=(10, 10))
sns.lineplot(x=list(range(1, EPOCHS + 1)), y=losses["training loss"])
sns.lineplot(x=list(range(1, EPOCHS + 1)), y=losses["validation loss"])
plt.savefig("images/losses")

with torch.no_grad():
    embeddings = parameters[0][test_set[0]]
    hidden_layer_output = torch.tanh(
        embeddings.view(-1, BLOCK_SIZE * EMBEDDING_LENGTH) @ parameters[1]
        + parameters[2]
    )
    logits = (
        embeddings.view(-1, BLOCK_SIZE * EMBEDDING_LENGTH) @ parameters[3]
        + hidden_layer_output @ parameters[4]
        + parameters[5]
    )
    test_loss = F.cross_entropy(logits, test_set[1])
print(f"Test Loss: {test_loss}")
wandb.log({"test_loss": test_loss.item()})
wandb.finish()
