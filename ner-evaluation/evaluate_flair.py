import json
import os

from typing import Text, Tuple, List, Dict

from flair.data import Corpus, Sentence, Token
from flair.embeddings import FlairEmbeddings
from nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from torch.utils.data import Dataset

from nlu.test import (
    get_eval_data,
    get_entity_extractors,
    evaluate_entities,
    remove_pretrained_extractors,
    EntityEvaluationResult,
)
from rasa.nlu import load_data
from rasa.nlu.model import Trainer, Interpreter
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData


class CustomDataset(Dataset):
    def __init__(self, sentences: List[Sentence]):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index: int = 0) -> Sentence:
        return self.sentences[index]


def load_training_data(
    path: Text, keep_split: bool = False, train_frac: float = 0.8
) -> Tuple[TrainingData, TrainingData]:
    if keep_split and os.path.isdir(path):
        data_train = load_data(os.path.join(path, "train.md"))
        data_test = load_data(os.path.join(path, "test.md"))
    else:
        training_data = load_data(path)
        data_train, data_test = training_data.train_test_split(train_frac)

    return data_train, data_test


def training_data_to_corpus(data_train: TrainingData):
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


def train_model(model_path: Text, data_train: TrainingData):
    from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
    from typing import List

    corpus = training_data_to_corpus(data_train)

    tag_type = "ner"

    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings("glove"),
        # FlairEmbeddings("news-forward"),
        # FlairEmbeddings("news-backward"),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
    )

    # 6. initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train(model_path, learning_rate=0.1, mini_batch_size=16, max_epochs=2)


def get_interpreter(data_train: TrainingData) -> Interpreter:
    config = RasaNLUModelConfig(
        configuration_values={
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "RegexFeaturizer"},
                {"name": "CRFEntityExtractor"},
            ]
        }
    )
    trainer = Trainer(config)
    return trainer.train(data_train)


def tokenize(text: Text):
    interpreter = Interpreter([WhitespaceTokenizer(), CRFEntityExtractor()], None)
    result = interpreter.parse(text)
    return result.get("tokens", []), result.get("entities", [])


def evaluate_model(
    model_path: Text, test_data: TrainingData, interpreter: Interpreter
) -> Dict:
    from flair.data import Sentence
    from flair.models import SequenceTagger

    tagger = SequenceTagger.load(os.path.join(model_path, "final-model.pt"))
    interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)

    entity_results = []

    for ex in test_data.training_examples:
        result = interpreter.parse(ex.text, only_output_properties=False)
        sentence = Sentence(result.get("text"), ex.text)
        tagger.predict(sentence)

        spans = sentence.get_spans("ner")

        predicted_entities = []
        for s in spans:
            predicted_entities.append(
                {
                    "value": s.text,
                    "start": s.start_pos,
                    "end": s.end_pos,
                    "entity": s.tag,
                    "extractor": "flair",
                    "confidence": s.score,
                }
            )

        entity_results.append(
            EntityEvaluationResult(
                ex.get("entities", []), predicted_entities, result.get("tokens", [])
            )
        )

    if entity_results:
        extractors = ["flair"]
        return evaluate_entities(entity_results, set(extractors), None)["flair"]

    return {}


def create_output_files(
    data_set: Text, result_folder: Text = "flair-results"
) -> Tuple[Text, Text, Text]:
    report_folder = os.path.join(result_folder, data_set)
    os.makedirs(report_folder, exist_ok=True)

    report_file = os.path.join(report_folder, "report.txt")
    if os.path.exists(report_file):
        os.remove(report_file)

    result_file = os.path.join(report_folder, "results.txt")
    if os.path.exists(result_file):
        os.remove(result_file)

    configuration_file = os.path.join(report_folder, "config.txt")
    if os.path.exists(configuration_file):
        os.remove(configuration_file)

    return report_file, result_file, configuration_file


def write_results(accuracy, f1_score, precision, result_file):
    f = open(result_file, "a")
    f.write("RESULTS\n")
    f.write("accuracy: {}\n".format(str(accuracy)))
    f.write("f1_score: {}\n".format(str(f1_score)))
    f.write("precision: {}\n".format(str(precision)))
    f.close()


def add_to_report(i, report, report_file):
    f = open(report_file, "a")
    f.write("#" * 100)
    f.write("\n")
    f.write("RUN {}\n\n".format(i))
    f.write(report)
    f.write("\n\n")
    f.close()


def run(data_path: Text, runs: int = 1, train_frac: float = 0.8):
    data_set = os.path.splitext(os.path.basename(data_path))[0]
    report_file, result_file, configuration_file = create_output_files(
        data_set, "flair-results"
    )

    accuracy_list = []
    f1_score_list = []
    precision_list = []

    for i in range(runs):
        data_train, data_test = load_training_data(data_path, train_frac=train_frac)
        interpreter = get_interpreter(data_train)

        train_model("flair-results/tagger", data_train)
        result = evaluate_model("flair-results/tagger", data_test, interpreter)

        report = result["report"]
        accuracy_list.append(result["accuracy"])
        f1_score_list.append(result["f1_score"])
        precision_list.append(result["precision"])

        add_to_report(i, report, report_file)

    accuracy = sum(accuracy_list) / len(accuracy_list)
    precision = sum(precision_list) / len(precision_list)
    f1_score = sum(f1_score_list) / len(f1_score_list)

    write_results(accuracy, f1_score, precision, result_file)

    f = open(configuration_file, "w")
    f.write("CONFIG\n")
    f.write("runs: {}\n".format(runs))
    f.write("training data: {}\n".format(train_frac))
    f.close()


if __name__ == "__main__":
    run("data/AddToPlaylist.json")
    run("data/BookRestaurant.json")
    run("data/GetWeather.json")
    run("data/RateBook.json")
    run("data/SearchCreativeWork.json")
    run("data/SearchScreeningEvent.json")
    run("data/BTC")
    run("data/re3d")
    run("data/WNUT17")
    run("data/Ritter.md")
