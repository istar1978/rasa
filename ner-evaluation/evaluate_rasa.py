import json
import os

from typing import Text, Tuple, List, Dict

from nlu.test import (
    get_eval_data,
    get_entity_extractors,
    evaluate_entities,
    remove_pretrained_extractors,
)
from rasa.nlu import load_data
from rasa.nlu.model import Trainer, Interpreter
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData


DEFAULT_PIPELINE = [
    {"name": "WhitespaceTokenizer"},
    {"name": "RegexFeaturizer"},
    {"name": "CRFEntityExtractor"},
]

SPACY_PIPELINE = [
    {"name": "SpacyNLP"},
    {"name": "SpacyTokenizer"},
    {"name": "SpacyFeaturizer"},
    {"name": "RegexFeaturizer"},
    {"name": "CRFEntityExtractor"},
]


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


def train_model(config: RasaNLUModelConfig, data_train: TrainingData) -> Interpreter:
    trainer = Trainer(config)
    return trainer.train(data_train)


def evaluate_model(interpreter: Interpreter, test_data: TrainingData) -> Dict:
    interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)

    result = {"entity_evaluation": None}

    _, entity_results = get_eval_data(interpreter, test_data)

    if entity_results:
        extractors = get_entity_extractors(interpreter)
        result["entity_evaluation"] = evaluate_entities(
            entity_results, extractors, None
        )

    return result["entity_evaluation"]["CRFEntityExtractor"]


def create_config(pipeline: List[Dict]) -> RasaNLUModelConfig:
    return RasaNLUModelConfig(configuration_values={"pipeline": pipeline})


def create_output_files(
    data_set: Text, result_folder: Text = "rasa-results"
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


def run(
    data_path: Text,
    runs: int = 5,
    pipeline: List[Dict] = DEFAULT_PIPELINE,
    train_frac: float = 0.8,
):
    data_set = os.path.splitext(os.path.basename(data_path))[0]
    report_file, result_file, configuration_file = create_output_files(data_set)

    config = create_config(pipeline)

    accuracy_list = []
    f1_score_list = []
    precision_list = []

    for i in range(runs):
        data_train, data_test = load_training_data(data_path, train_frac=train_frac)

        interpreter = train_model(config, data_train)
        result = evaluate_model(interpreter, data_test)

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
    f.write("pipeline: {}".format(pipeline))
    f.write("runs: {}".format(runs))
    f.write("training data: {}".format(train_frac))
    f.close()


if __name__ == "__main__":
    run("data/AddToPlaylist.json")
    run("data/BookRestaurant.json")
    run("data/GetWeather.json")
    run("data/RateBook.json")
    run("data/SearchCreativeWork.json")
    run("data/SearchScreeningEvent.json")
    run("data/BTC")
    run("data/redis")
    run("data/WNUT17")
