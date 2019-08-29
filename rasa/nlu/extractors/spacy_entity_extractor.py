import random
import re
import typing
from typing import Any, Dict, List, Text, Optional

import spacy
from spacy.util import minibatch, compounding

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import TrainingData
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.training_data import Message

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc  # pytype: disable=import-error


class SpacyEntityExtractor(EntityExtractor):
    provides = ["entities"]

    requires = []

    defaults = {
        # by default all dimensions recognized by spacy are returned
        # dimensions can be configured to contain an array of strings
        # with the names of the dimensions to filter for
        "dimensions": None,
        # The maximum number of iterations for optimization algorithms.
        "n_iter": 50,
        # Dropout to use during training.
        "dropout": 0.5,
    }

    def __init__(self, component_config: Text = None) -> None:
        self.spacy_nlp = None

        super(SpacyEntityExtractor, self).__init__(component_config)

    def process(self, message: Message, **kwargs: Any) -> None:
        # can't use the existing doc here (spacy_doc on the message)
        # because tokens are lower cased which is bad for NER

        doc = self.spacy_nlp(message.text)
        all_extracted = self.add_extractor_name(self.extract_entities(doc))
        dimensions = self.component_config["dimensions"]
        extracted = SpacyEntityExtractor.filter_irrelevant_entities(
            all_extracted, dimensions
        )
        message.set(
            "entities", message.get("entities", []) + extracted, add_to_output=True
        )

    @staticmethod
    def extract_entities(doc: "Doc") -> List[Dict[Text, Any]]:
        entities = [
            {
                "entity": ent.label_,
                "value": ent.text,
                "start": ent.start_char,
                "confidence": None,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]
        return entities

    import re

    def trim_entity_spans(self, data: list) -> list:
        """Removes leading and trailing white spaces from entity spans.

        Args:
            data (list): The data to be cleaned in spaCy JSON format.

        Returns:
            list: The cleaned data.
        """
        invalid_span_tokens = re.compile(r"\s")

        cleaned_data = []
        for text, annotations in data:
            entities = annotations["entities"]
            valid_entities = []
            for start, end, label in entities:
                valid_start = start
                valid_end = end
                while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]
                ):
                    valid_start += 1
                while valid_end > 1 and invalid_span_tokens.match(text[valid_end - 1]):
                    valid_end -= 1
                valid_entities.append([valid_start, valid_end, label])
            cleaned_data.append([text, {"entities": valid_entities}])

        return cleaned_data

    def convert_to_spacy_format(self, training_data):
        examples = []

        for example in training_data.training_examples:
            text = example.text
            entities = [
                (entity["start"], entity["end"], entity["entity"])
                for entity in example.get("entities", [])
            ]

            examples.append((text, {"entities": entities}))

        return self.trim_entity_spans(examples)

    def train(
        self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs: Any
    ) -> None:
        self.spacy_nlp = spacy.blank("en")

        data = self.convert_to_spacy_format(training_data)

        if "ner" not in self.spacy_nlp.pipe_names:
            ner = self.spacy_nlp.create_pipe("ner")
            self.spacy_nlp.add_pipe(ner, last=True)
        else:
            ner = self.spacy_nlp.get_pipe("ner")

        # add labels
        for _, annotations in data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in self.spacy_nlp.pipe_names if pipe != "ner"]
        with self.spacy_nlp.disable_pipes(*other_pipes):  # only train NER
            self.spacy_nlp.begin_training()
            for itn in range(self.component_config["n_iter"]):
                random.shuffle(data)
                losses = {}
                batches = minibatch(data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.spacy_nlp.update(
                        texts,
                        annotations,
                        drop=self.component_config["dropout"],
                        losses=losses,
                    )

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component: Optional["SpacyEntityExtractor"] = None,
        **kwargs: Any
    ) -> "SpacyEntityExtractor":
        nlp = spacy.load(model_dir)
        cls(nlp)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        self.spacy_nlp.to_disk(model_dir)
