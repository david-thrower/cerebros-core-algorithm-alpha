



import nltk
import numpy as np
import re

# Text Sample: Class for a text sample


class TextSample:
    def __init__(self, prompt: str, response: str):
        self.prompt = prompt
        self.response = response

# Example data
sample1 = TextSample(prompt="Tell me all about the capitol of France",
                     response="Paris is known as the city of love")
sample2 = TextSample(prompt="Write a haiku about life",
                     response="Life blows.\nYou go to school.\nYou go to work")
sample3 = TextSample(prompt="Write an ode to Silence:",
                     response="Silence is awesome. Silence is rare. Silence is beauty. Silence is nowhere.")
samples = [sample1, sample2, sample3]

# Empty list (may want to change to a dict for scalability) 


def split_string(text: str) -> list:
    try:
        words = nltk.word_tokenize(text)
    except LookupError as err:
        print(f"Looks like punkt is missing: \n {err} \n "
              "Downloading punkt to try resolving this:")
        nltk.download('punkt')
        words = nltk.word_tokenize(text)

    return words


def create_data_and_labels(samples):
    data = []
    labels = []
    for i in np.arange(len(samples)):
        sample_0 = samples[i]
        prompt_0_str = sample_0.prompt
        response_0_str = sample_0.response

        response_0_list = split_string(response_0_str)

        data_0 = []
        label_0 = []
        data_0 = prompt_0_str
        for j in np.arange(len(response_0_list)):
            if i == 0:
                label_0 = response_0_list[j]
            else:
                data_0 += f" {response_0_list[j - 1]}"
                label_0 = response_0_list[j]
            data.append(data_0)
            labels.append(label_0)
