from io import open
import os
from conllu import parse_incr


DATA_DIRECTORY = "data"
CONLL_FILEDS = ("form", "entity")  # form = text


def convert(input, output):
    print ("Parsing file '{}'.".format(input))

    if os.path.exists(output):
        os.remove(output)

    data_file = open(input, "r", encoding="utf-8")

    f = open(output, "a")
    f.write("## intent:ner_examples")
    f.write("\n")
    f.close()

    for tokenlist in parse_incr(data_file, fields=CONLL_FILEDS):

        tokens = []
        entity = None
        found_entity = False

        for token in tokenlist:
            if "entity" not in token:
                token["entity"] = "O"

            # new entity found
            if token["entity"].startswith("B-") and not found_entity:
                tokens.append("[{}".format(token["form"]))
                found_entity = True
                entity = token["entity"][2:]

            # new entity directly after another entity
            elif token["entity"].startswith("B-") and found_entity:
                last_token = tokens[-1]
                del tokens[-1]
                tokens.append("{}]({})".format(last_token, entity))
                tokens.append("[{}".format(token["form"]))
                found_entity = True
                entity = token["entity"][2:]

            # entity over
            elif found_entity and token["entity"] == "O":
                last_token = tokens[-1]
                del tokens[-1]
                tokens.append("{}]({})".format(last_token, entity))
                found_entity = False
                tokens.append(token["form"])

            else:
                tokens.append(token["form"])

        if found_entity:
            last_token = tokens[-1]
            del tokens[-1]
            tokens.append("{}]({})".format(last_token, entity))

        text = " ".join(tokens)

        f = open(output, "a")
        f.write("  - ")
        f.write(text)
        f.write("\n")
        f.close()


def process():
    from pathlib import Path

    pathlist = Path(DATA_DIRECTORY).glob("**/*.conll")
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path)
        convert(path_in_str, path_in_str + ".md")


if __name__ == "__main__":
    process()
