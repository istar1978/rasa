import os
import tempfile
from typing import Any, Dict, List, Text

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.training_data import Message
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Corpus, Sentence, FlairDataset
from flair.embeddings import (
    FlairEmbeddings,
    TokenEmbeddings,
    WordEmbeddings,
    StackedEmbeddings,
)


class CustomDataset(FlairDataset):
    def __init__(self, sentences: List[Sentence]):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index: int = 0) -> Sentence:
        return self.sentences[index]


class FlairEntityExtractor(EntityExtractor):
    provides = ["entities"]

    requires = []

    defaults = {
        "hidden_size": 256,
        "learning_rate": 0.1,
        "mini_batch_size": 16,
        "max_epochs": 2,
        "use_glove_embeddings": True,
        "use_flair_embeddings": True,
    }

    def __init__(self, component_config: Text = None) -> None:
        self.model_path = tempfile.mkdtemp()
        self.tagger = None

        super(FlairEntityExtractor, self).__init__(component_config)

    def extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Take a sentence and return entities in json format"""
        sentence = Sentence(message.text)

        if not self.tagger:
            self.tagger = SequenceTagger.load(
                os.path.join(self.model_path, "final-model.pt")
            )

        self.tagger.predict(sentence)

        spans = sentence.get_spans("ner")
        return [
            {
                "value": s.text,
                "start": s.start_pos,
                "end": s.end_pos,
                "entity": s.tag,
                "extractor": "flair",
                "confidence": s.score,
            }
            for s in spans
        ]

    def process(self, message: Message, **kwargs: Any) -> None:
        extracted = self.add_extractor_name(self.extract_entities(message))
        message.set(
            "entities", message.get("entities", []) + extracted, add_to_output=True
        )

    def convert_to_flair_format(self, data_train):
        sentences = []

        for ex in data_train.training_examples:
            sentence = Sentence(ex.text)
            for token in sentence.tokens:
                for entity in ex.get("entities"):
                    if (
                        token.start_pos >= entity["start"]
                        and token.end_pos <= entity["end"]
                    ):
                        token.add_tag("ner", entity["entity"])

            sentences.append(sentence)

        return Corpus(
            train=CustomDataset(sentences),
            dev=CustomDataset(sentences),
            test=CustomDataset([]),
        )

    def train(
        self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs: Any
    ) -> None:
        corpus = self.convert_to_flair_format(training_data)

        tag_type = "ner"

        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

        embedding_types: List[TokenEmbeddings] = []

        if self.component_config["use_glove_embeddings"]:
            embedding_types.append(WordEmbeddings("glove"))

        if self.component_config["use_flair_embeddings"]:
            embedding_types.append(FlairEmbeddings("news-forward"))
            embedding_types.append(FlairEmbeddings("news-backward"))

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        tagger: SequenceTagger = SequenceTagger(
            hidden_size=self.component_config["hidden_size"],
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=tag_type,
            use_crf=True,
        )

        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        trainer.train(
            self.model_path,
            learning_rate=self.component_config["learning_rate"],
            mini_batch_size=self.component_config["mini_batch_size"],
            max_epochs=self.component_config["max_epochs"],
        )

        self.tagger = SequenceTagger.load(
            os.path.join(self.model_path, "final-model.pt")
        )
