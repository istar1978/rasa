import os
import random

from typing import Text

from spellchecker import SpellChecker

from nlu import load_data
from rasa.nlu.training_data import TrainingData


def load_training_data(path: Text) -> TrainingData:
    return load_data(path)


def misspell(word: Text):
    spell_checker = SpellChecker()

    candidates = spell_checker.edit_distance_1(word)
    candidates = [
        (c, spell_checker.word_probability(c))
        for c in candidates
        if spell_checker.word_probability(c) > 0.0
    ]
    candidates.sort(key=lambda tup: tup[1])

    if candidates and len(candidates) > 1:
        index = random.randint(0, 1)
        return candidates[index][0]

    if candidates:
        return candidates[0][0]

    return word


def run(data_path: Text):
    data_set = os.path.splitext(os.path.basename(data_path))[0]
    out_folder = os.path.join("data", "new", data_set)
    out_file = os.path.join(out_folder, "data.md")
    os.makedirs(out_folder, exist_ok=True)

    f = open(out_file, "w")
    f.write("## intent:examples")
    f.write("\n")
    f.close()

    data = load_training_data(data_path)

    for ex in data.training_examples:
        text = ex.text
        entities = ex.get("entities", [])
        tokens = text.split()

        new_tokens = []
        offset = 0
        entity_index = 0
        entity = entities[entity_index] if entities else None
        start_entity = False
        end_entity = False
        entity_value = None
        entity_tag = None

        for token in tokens:
            new_token = token

            if entity and offset == entity["start"]:
                start_entity = True
            if entity and offset >= entity["end"]:
                end_entity = True
                entity_value = entity["value"]
                entity_tag = entity["entity"]
                entity_index += 1

            entity = (
                entities[entity_index]
                if entities and entity_index < len(entities)
                else None
            )

            if entity and entity["start"] <= offset <= entity["end"]:
                if random.randint(0, 3) == 1:
                    new_token = misspell(token)
            elif random.randint(0, 10) == 5:
                new_token = misspell(token)

            offset += len(token) + 1

            if start_entity:
                new_token = "[" + new_token
                start_entity = False
            if end_entity:
                new_tokens[-1] = "{}]({}:{})".format(
                    new_tokens[-1], entity_tag, entity_value
                )
                entity_value = None
                end_entity = False

            new_tokens.append(new_token)

        text = " ".join(new_tokens)

        f = open(out_file, "a")
        f.write("- ")
        f.write(text)
        f.write("\n")
        f.close()


if __name__ == "__main__":
    run("data/WNUT17")
