"""This is an implementation of a trigram model"""
# pylint: disable=invalid-name
import random
import math
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F


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
    model_weights: torch.Tensor,
    training_set: Tuple[torch.Tensor],
    val_set: Tuple[torch.Tensor],
    reg_value: float,
    epochs: int,
    learning_rate: float,
) -> Tuple[torch.Tensor, dict]:
    """
    Function to train our classifier

    Args:
        model_weights (torch.Tensor): Weights used in our classes
        training_set (Tuple[torch.Tensor]): dataset used to train classifier
        as a tuple of the features and labels i.e features, labels
        val_set (Tuple[torch.Tensor]): dataset used to validate classifier
        as a tuple of the features and labels i.e features, labels
        reg_value (float): strength of the regularization to be used
        epochs (int): Number of iterations to train for
        learning_rate (float): Value controlling the rate of change of weights
          at each epoch

    Returns:
        Tuple[torch.Tensor, dict]: A tuple of the models weights and a dictionary
        containing both training and validations losses recorded during training.
    """
    loss = {"training losses": [], "validation losses": []}
    for epoch in range(1, epochs + 1):
        logit = model_weights[training_set[0]]
        train_loss = (
            F.cross_entropy(logit, training_set[1])
            + reg_value * (model_weights**2).mean()
        )
        loss["training losses"].append(train_loss.item())
        model_weights.grad = None
        train_loss.backward()
        model_weights.data += -learning_rate * model_weights.grad
        with torch.no_grad():
            logit = model_weights[
                val_set[0]
            ]
            valid_loss = (
                F.cross_entropy(logit, val_set[1])
            )
            loss["validation losses"].append(valid_loss.item())
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} training loss {train_loss}, validation loss {valid_loss}"
            )
    return model_weights, loss


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with open("names.txt", "r", encoding="utf-8") as file:
    data = file.read().splitlines()
SPECIAL_TOKEN = "<>"

unique_characters = sorted(list(set("".join(data))))
unique_characters = [SPECIAL_TOKEN] + unique_characters
unique_inputs = [x + y for x in unique_characters for y in unique_characters]
inputs_to_index = {input_: index for index, input_ in enumerate(unique_inputs)}
character_to_index = {
    character: index for index, character in enumerate(unique_characters)
}
index_to_character = {
    index: character for index, character in enumerate(unique_characters)
}

features = []
labels = []
for word in data:
    characters = [SPECIAL_TOKEN] * 2 + list(word) + [SPECIAL_TOKEN]
    for character_1, character_2, character_3 in zip(
        characters[:-2], characters[1:-1], characters[2:]
    ):
        input_ = character_1 + character_2
        input_index = inputs_to_index[input_]
        character_3_index = character_to_index[character_3]
        features.append(input_index)
        labels.append(character_3_index)

features = torch.tensor(features, device=DEVICE)
labels = torch.tensor(labels, device=DEVICE)
train_set, validation_set, test_set = split_data(features, labels)

# training
EPOCHS = 100
LEARNING_RATE = 100
regularization_values = [10**i for i in range(-5, 3)]
final_val_losses = []
for regularization_value in regularization_values:
    generator = torch.Generator(device=DEVICE).manual_seed(12000)
    weights = torch.randn(
        (len(unique_inputs), len(unique_characters)),
        generator=generator,
        device=DEVICE,
        requires_grad=True,
    )
    print(f"\nUsing {regularization_value}....")
    _, losses = train(
        model_weights=weights,
        training_set=train_set,
        val_set=validation_set,
        reg_value=regularization_value,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
    )
    final_val_losses.append(losses["validation losses"][-1])

plt.figure(figsize=(15, 15))
sns.lineplot(
    x=[round(math.log(i, 10)) for i in regularization_values],
    y=final_val_losses,
)
plt.savefig("images/regularization_tuning")

reg_val_index = final_val_losses.index(min(final_val_losses))
regularization_value = regularization_values[reg_val_index]
print(f"\nBest regularization Value is {regularization_value}\n")

generator = torch.Generator(device=DEVICE).manual_seed(12000)
weights = torch.randn(
    (len(unique_inputs), len(unique_characters)),
    generator=generator,
    device=DEVICE,
    requires_grad=True,
)
EPOCHS = 1000

weights, losses = train(
    model_weights=weights,
    training_set=train_set,
    val_set=validation_set,
    reg_value=regularization_value,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
)

plt.figure(figsize=(15, 15))
sns.lineplot(x=list(range(1, EPOCHS + 1)), y=losses["training losses"])
sns.lineplot(x=list(range(1, EPOCHS + 1)), y=losses["validation losses"])
plt.savefig("images/losses")

with torch.no_grad():
    logits = weights[test_set[0]]
    test_loss = (
        F.cross_entropy(logits, test_set[1])
    )
    print(f"Test loss: {test_loss.item()}")

# inference
generated_string = ""
input_index = 0
accumulated_loss = 0
with torch.no_grad():
    while True:
        input_tensor = torch.tensor([input_index], device=DEVICE)
        logits = weights[input_tensor]
        trigram_counts = logits.exp()
        trigram_probabilities = trigram_counts / trigram_counts.sum()
        character_index = torch.multinomial(
            input=trigram_probabilities,
            num_samples=1,
            replacement=True,
        ).item()
        inference_loss = F.cross_entropy(logits,torch.tensor([character_index], device=DEVICE))
        accumulated_loss += inference_loss
        if character_index == 0:
            break
        generated_string += index_to_character[character_index]
        if len(generated_string) < 2:
            next_input = SPECIAL_TOKEN + generated_string
        else:
            next_input = generated_string[-2:]
        input_index = inputs_to_index[next_input]

print(generated_string)
print(accumulated_loss.item() / (len(generated_string) + 1))
