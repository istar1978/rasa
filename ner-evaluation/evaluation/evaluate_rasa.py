import os

from typing import Text, List, Dict

from nlu.test import (
    get_eval_data,
    get_entity_extractors,
    evaluate_entities,
    remove_pretrained_extractors,
)
from rasa.nlu.model import Trainer, Interpreter
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData

from utils import (
    create_output_files,
    load_training_data,
    add_to_report,
    write_results,
    write_config,
)

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


def train_model(pipeline: List[Dict], data_train: TrainingData) -> Interpreter:
    config = RasaNLUModelConfig(configuration_values={"pipeline": pipeline})
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


def run(
    data_path: Text,
    runs: int = 1,
    pipeline: List[Dict] = DEFAULT_PIPELINE,
    train_frac: float = 0.8,
):
    data_set = os.path.splitext(os.path.basename(data_path))[0]
    report_file, result_file, configuration_file = create_output_files(
        data_set, result_folder="results/rasa/"
    )

    accuracy_list = []
    f1_score_list = []
    precision_list = []

    for i in range(runs):
        data_train, data_test = load_training_data(data_path, train_frac=train_frac)

        interpreter = train_model(pipeline, data_train)
        result = evaluate_model(interpreter, data_test)

        report = result["report"]
        accuracy_list.append(result["accuracy"])
        f1_score_list.append(result["f1_score"])
        precision_list.append(result["precision"])

        add_to_report(report_file, i, report)

    accuracy = sum(accuracy_list) / len(accuracy_list)
    precision = sum(precision_list) / len(precision_list)
    f1_score = sum(f1_score_list) / len(f1_score_list)

    write_results(result_file, accuracy, f1_score, precision)
    write_config(configuration_file, runs, train_frac, pipeline)


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
