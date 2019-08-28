import random
import typing
from typing import Any, Dict, List, Text, Optional

import spacy
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
        "dimensions": None
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

    def convert_to_spacy_format(self, training_data):
        examples = []

        for example in training_data.training_examples:
            text = example.text
            entities = [
                (entity["start"], entity["end"], entity["entity"])
                for entity in example.get("entities", [])
            ]

            examples.append((text, {"entities": entities}))

        return examples

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
            optimizer = self.spacy_nlp.begin_training()
            for itn in range(10):
                random.shuffle(data)
                losses = {}
                for text, annotations in data:
                    self.spacy_nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.6,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
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
