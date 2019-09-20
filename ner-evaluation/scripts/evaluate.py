import traceback
import json
import os
from typing import Tuple, Optional, Text, List, Dict

from rasa.nlu.test import (
    get_eval_data,
    get_entity_extractors,
    evaluate_entities,
    remove_pretrained_extractors,
)
from rasa.nlu.model import Trainer, Interpreter
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData
from rasa.nlu.training_data.loading import load_files
from rasa.utils.common import set_log_level
from rasa.utils.io import list_files


DEFAULT_PIPELINE = [{"name": "WhitespaceTokenizer"}, {"name": "CRFEntityExtractor"}]

SPACY_PIPELINE = [
    {"name": "SpacyNLP"},
    {"name": "SpacyTokenizer"},
    {
        "name": "CRFEntityExtractor",
        "features": [
            ["low", "title", "upper"],
            [
                "bias",
                "low",
                "prefix5",
                "prefix2",
                "suffix5",
                "suffix3",
                "suffix2",
                "upper",
                "title",
                "digit",
                "pattern",
                "pos",
                "pos2",
            ],
            ["low", "title", "upper"],
        ],
    },
]

SPACY_NER_PIPELINE = [{"name": "WhitespaceTokenizer"}, {"name": "SpacyEntityExtractor"}]

FLAIR_PIPELINE = [{"name": "WhitespaceTokenizer"}, {"name": "FlairEntityExtractor"}]

TF_PIPELINE = [
    {"name": "WhitespaceTokenizer"},
    {"name": "CountVectorsFeaturizer"},
    {"name": "TensorflowCrfEntityExtractor"},
]


def load_training_data(
    path: Text, train_frac: float = 0.8, typo: bool = False
) -> Tuple[TrainingData, TrainingData]:

    files = list_files(path)

    files = [f for f in files if (f.endswith(".md") or f.endswith(".json"))]
    if typo:
        files = [f for f in files if os.path.basename(f).startswith("typo_")]
    else:
        files = [f for f in files if not os.path.basename(f).startswith("typo_")]

    train_test_split = [os.path.basename(f) == "test.md" for f in files].count(
        True
    ) == 1

    if train_test_split:
        data_train = load_files([f for f in files if "train" in f])
        data_test = load_files([f for f in files if "test" in f])
    else:
        training_data = load_files(files)
        data_train, data_test = training_data.train_test_split(train_frac)

    return data_train, data_test


def create_output_files(
    data_set: Text,
    result_folder: Text = "results",
    runs: int = 5,
    train_frac: float = 0.8,
    typo: bool = False,
) -> Tuple[Text, Text, Text, Text]:
    name = "{}x{}x{}x{}".format(data_set, typo, train_frac, runs)

    report_folder = os.path.join(result_folder, name)
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

    return report_folder, report_file, result_file, configuration_file


def write_results(
    result_file: Text, accuracy: float, f1_score: float, precision: float
):
    f = open(result_file, "a")
    f.write("RESULTS\n")
    f.write("accuracy: {}\n".format(str(accuracy)))
    f.write("f1_score: {}\n".format(str(f1_score)))
    f.write("precision: {}\n".format(str(precision)))
    f.close()


def add_to_report(report_file: Text, run: int, report: Dict):
    f = open(report_file, "a")
    f.write("#" * 100)
    f.write("\n")
    f.write("RUN {}\n\n".format(run))
    f.write(json.dumps(report, indent=4))
    f.write("\n\n")
    f.close()


def write_config(
    configuration_file: Text,
    runs: int,
    train_frac: float,
    pipeline: Optional[List[Dict]] = None,
):
    f = open(configuration_file, "w")
    f.write("CONFIG\n")
    f.write("pipeline: {}\n".format(pipeline))
    f.write("runs: {}\n".format(runs))
    f.write("training data: {}\n".format(train_frac))
    f.close()


def train_model(pipeline: List[Dict], data_train: TrainingData) -> Interpreter:
    config = RasaNLUModelConfig(configuration_values={"pipeline": pipeline})
    trainer = Trainer(config)
    return trainer.train(data_train)


def evaluate_model(
    interpreter: Interpreter, test_data: TrainingData, result_folder: Text
) -> Dict:
    interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)

    result = {"entity_evaluation": None}

    _, _, entity_results = get_eval_data(interpreter, test_data)

    extractors = get_entity_extractors(interpreter)

    if entity_results:
        result["entity_evaluation"] = evaluate_entities(
            entity_results, extractors, output_directory=result_folder, errors=True
        )

    return result["entity_evaluation"][extractors.pop()]


def run(
    data_path: Text,
    runs: int,
    pipeline: List[Dict],
    pipeline_name: Text,
    train_frac: float,
    typo: bool,
    output_folder: Text,
):
    set_log_level(10)

    print (
        "Evaluating pipeline '{}' on dataset '{}' (typo: {}, train_frac: {}, runs: {}).".format(
            pipeline_name, data_path, typo, train_frac, runs
        )
    )

    data_set = os.path.basename(data_path)
    result_folder = os.path.join(output_folder, pipeline_name)
    report_folder, report_file, result_file, configuration_file = create_output_files(
        data_set, result_folder, runs, train_frac, typo
    )

    accuracy_list = []
    f1_score_list = []
    precision_list = []

    for i in range(runs):
        data_train, data_test = load_training_data(
            data_path, train_frac=train_frac, typo=typo
        )

        interpreter = train_model(pipeline, data_train)

        result = evaluate_model(interpreter, data_test, report_folder)

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

    print (
        "Done evaluating pipeline '{}' on dataset '{}'. Results written to '{}'.".format(
            pipeline_name, data_path, report_folder
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Named-Entity-Recognition Evaluation")
    parser.add_argument("data", type=str, help="path to dataset folder")
    parser.add_argument("--output", type=str, default="local", help="output folder")
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["default", "default-spacy", "spacy", "flair", "tensorflow"],
        default="tensorflow",
        help="pipeline to evaluate",
    )
    parser.add_argument("--typo", action="store_true", help="use typo dataset")
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.5,
        help="percentage of training datset examples",
    )
    parser.add_argument("--runs", type=int, default=1, help="number of runs")

    args = parser.parse_args()

    pipelines = {
        "default": (DEFAULT_PIPELINE, "default_pipeline"),
        "default-spacy": (SPACY_PIPELINE, "default_spacy_pipeline"),
        "spacy": (SPACY_NER_PIPELINE, "spacy_pipeline"),
        "flair": (FLAIR_PIPELINE, "flair_pipeline"),
        "tensorflow": (TF_PIPELINE, "tf_pipeline"),
    }

    pipeline = pipelines[args.pipeline][0]
    pipeline_name = pipelines[args.pipeline][1]

    run(
        args.data,
        pipeline=pipeline,
        pipeline_name=pipeline_name,
        typo=args.typo,
        train_frac=args.train_frac,
        output_folder=args.output,
        runs=args.runs,
    )
