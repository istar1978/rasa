import json
from io import open
import os
from conllu import parse_incr


DATA_DIRECTORY = "/Users/tabergma/Repositories/entity-recognition-datasets/data/Coached-Conversational-Preference-Elicitation"


def generate_entity_md(text, entity):
    """generates markdown for an entity object."""
    entity_text = text[entity["start"] : entity["end"]]
    entity_type = entity["entity"]
    if entity_text != entity["value"]:
        # add synonym suffix
        entity_type += ":{}".format(entity["value"])

    return "[{}]({})".format(entity_text, entity_type)


def merge_text_and_entities(text, entities):
    if not entities:
        return text

    md = ""
    entities = sorted(entities, key=lambda k: k["start"])

    pos = 0
    for entity in entities:
        md += text[pos : entity["start"]]
        md += generate_entity_md(text, entity)
        pos = entity["end"]

    md += text[pos:]

    return md


def get_entities(segments):
    entities = []
    for segment in segments:

        candidates = [
            s for s in segment["annotations"] if s["annotationType"] == "ENTITY_NAME"
        ]

        if len(candidates) == 1:
            entities.append(
                {
                    "start": segment["startIndex"],
                    "end": segment["endIndex"],
                    "value": segment["text"],
                    "entity": candidates[0]["entityType"],
                }
            )
    return entities


def convert(input, output):
    print ("Parsing file '{}'.".format(input))

    if os.path.exists(output):
        os.remove(output)

    f = open(output, "a")
    f.write("## intent:ner_examples")
    f.write("\n")
    f.close()

    with open(input) as json_file:
        data = json.load(json_file)
        for conversation in data:
            for utterance in conversation["utterances"]:
                text = utterance["text"]
                entities = []
                if "segments" in utterance:
                    entities = get_entities(utterance["segments"])

                text = merge_text_and_entities(text, entities)

                f = open(output, "a")
                f.write("  - ")
                f.write(text)
                f.write("\n")
                f.close()


def process():
    from pathlib import Path

    pathlist = Path(DATA_DIRECTORY).glob("**/*.json")
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path)
        convert(path_in_str, path_in_str + ".md")


if __name__ == "__main__":
    process()
