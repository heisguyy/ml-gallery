"""
The script is the implementation of a simple probalistic bigram language model.
This is a code along with Andrej Karpathy's zero to hero neural network series.
https://www.youtube.com/watch?v=PaCmpygFfXo&t=3740s
"""
import torch
import torch.nn.functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

data = open("names.txt", "r", encoding="utf-8").read().splitlines()
SPECIAL_TOKEN = "<>"

unique_characters = sorted(list(set("".join(data))))
unique_characters = [SPECIAL_TOKEN] + unique_characters
character_to_index = {
    character: index for index, character in enumerate(unique_characters)
}
index_to_character = {
    index: character for index, character in enumerate(unique_characters)
}

features = []
labels = []
for word in data:
    character = [SPECIAL_TOKEN] + list(word) + [SPECIAL_TOKEN]
    for character_1, character_2 in zip(character[:-1], character[1:]):
        character_1_index = character_to_index[character_1]
        character_2_index = character_to_index[character_2]
        features.append(character_1_index)
        labels.append(character_2_index)
features = torch.tensor(features, device=DEVICE)
labels = torch.tensor(labels, device=DEVICE)
encoded_features = F.one_hot(features, num_classes=len(unique_characters)).float()

# training
LEARNING_RATE = 50
generator = torch.Generator(device=DEVICE).manual_seed(12000)
weights = torch.randn(
    (len(unique_characters), len(unique_characters)),
    device=DEVICE,
    generator=generator,
    requires_grad=True,
)
EPOCHS = 200
for epoch in range(1, EPOCHS + 1):
    logits = encoded_features @ weights
    bigram_count = logits.exp()
    bigram_probabilities = bigram_count / bigram_count.sum(dim=1, keepdim=True)
    # 0.01*(weights**2).mean() is regularization, equivalent
    # to the smoothening done in prob_bigram_lm.py
    loss = (
        -bigram_probabilities[torch.arange(len(labels), device=DEVICE), labels]
        .log()
        .mean()
        + 0.01 * (weights**2).mean()
    )
    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item()}")
    weights.grad = None
    loss.backward()
    weights.data -= LEARNING_RATE * weights.grad


# inference
generated_string = ""  # pylint: disable=invalid-name
generator = torch.Generator(device=DEVICE)
character_index = torch.tensor([0], device=DEVICE)  # pylint: disable=invalid-name
accumulated_loss = 0  # pylint: disable=invalid-name
with torch.no_grad():
    while True:
        encoded_character = F.one_hot(
            torch.tensor([character_index], device=DEVICE),
            num_classes=len(unique_characters),
        ).float()
        logits = encoded_character @ weights
        bigram_count = logits.exp()
        bigram_probabilities = bigram_count / bigram_count.sum(dim=1, keepdim=True)
        character_index = torch.multinomial(
            bigram_probabilities, 1, replacement=True, generator=generator
        ).item()
        loss = (
            -bigram_probabilities[0, character_index].log()
            + 0.01 * (weights**2).mean()
        )
        accumulated_loss += loss
        if character_index == 0:
            break
        generated_string += index_to_character[character_index]
print(generated_string)
print(accumulated_loss.item() / (len(generated_string) + 1))
