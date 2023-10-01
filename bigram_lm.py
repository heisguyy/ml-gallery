"""
The script is the implementation of a simple probalistic bigram language model.
This is a code along with Andrej Karpathy's zero to hero neural network series.
https://www.youtube.com/watch?v=PaCmpygFfXo&t=3740s
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def plot_heatmap(tensor: torch.Tensor, save_path: str) -> None:
    """
    Plots a heatmap of the given data and saves it to the given path.
    """
    plt.figure(figsize=(30, 30))
    axes = sns.heatmap(
        tensor.cpu(),
        cmap="Greys_r",
        fmt="d",
        cbar=False,
    )
    for i in range(len(unique_characters)):
        for j in range(len(unique_characters)):
            axes.text(
                j + 0.5,
                i + 0.5,
                f"{index_to_character[i]}{index_to_character[j]}",
                ha="center",
                va="bottom",
                color="w",
            )
            axes.text(
                j + 0.5,
                i + 0.5,
                f"{bigram_count[i, j].item():.4f}",
                ha="center",
                va="top",
                color="w",
            )
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_xlabel("Second Character")
    axes.set_ylabel("First Character")
    plt.savefig(save_path)


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

bigram_count = torch.zeros(
    (len(unique_characters), len(unique_characters)), device=DEVICE, dtype=torch.int16
)
for word in data:
    character = [SPECIAL_TOKEN] + list(word) + [SPECIAL_TOKEN]
    for character_1, character_2 in zip(character[:-1], character[1:]):
        character_1_index = character_to_index[character_1]
        character_2_index = character_to_index[character_2]
        bigram_count[character_1_index, character_2_index] += 1
plot_heatmap(bigram_count, "images/bigram_count.png")

# Add one to the count of everything to smoothen the probabilities,
# so we don't have 0 in the probability, log(0) is -inf.
bigram_count += 1

bigram_probabilities = bigram_count / bigram_count.sum(dim=1, keepdim=True)
plot_heatmap(bigram_probabilities, "images/bigram_probabilities.png")

generated_string = ""  # pylint: disable=invalid-name
generator = torch.Generator(device=DEVICE).manual_seed(1500)
character_index = 0  # pylint: disable=invalid-name
negative_log_likelihood = 0  # pylint: disable=invalid-name
while True:
    probabilities = bigram_probabilities[character_index]
    character_index = torch.multinomial(
        probabilities, 1, replacement=True, generator=generator
    ).item()
    log_character_probability = torch.log(probabilities[character_index])
    negative_log_likelihood -= log_character_probability
    if character_index == 0:
        break
    generated_string += index_to_character[character_index]
print(generated_string)
print(negative_log_likelihood.item() / (len(generated_string) + 1))
