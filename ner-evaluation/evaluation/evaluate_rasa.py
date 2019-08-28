import os
import traceback

from typing import Text, List, Dict

from rasa.nlu.test import (
    get_eval_data,
    get_entity_extractors,
    evaluate_entities,
    remove_pretrained_extractors,
)
from rasa.nlu.model import Trainer, Interpreter
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData
from rasa.utils.common import set_log_level

from utils import (
    create_output_files,
    load_training_data,
    add_to_report,
    write_results,
    write_config,
)

DEFAULT_PIPELINE = [{"name": "WhitespaceTokenizer"}, {"name": "CRFEntityExtractor"}]

SPACY_PIPELINE = [
    {"name": "SpacyNLP"},
    {"name": "SpacyTokenizer"},
    {"name": "SpacyFeaturizer"},
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


def train_model(pipeline: List[Dict], data_train: TrainingData) -> Interpreter:
    config = RasaNLUModelConfig(configuration_values={"pipeline": pipeline})
    trainer = Trainer(config)
    return trainer.train(data_train)


def evaluate_model(
    interpreter: Interpreter, test_data: TrainingData, result_folder: Text
) -> Dict:
    interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)

    result = {"entity_evaluation": None}

    _, entity_results = get_eval_data(interpreter, test_data)

    extractors = get_entity_extractors(interpreter)

    if entity_results:
        result["entity_evaluation"] = evaluate_entities(
            entity_results, extractors, output_directory=result_folder, errors=True
        )

    return result["entity_evaluation"][extractors.pop()]


def run(
    data_path: Text,
    runs: int = 5,
    pipeline: List[Dict] = DEFAULT_PIPELINE,
    pipeline_name: Text = "default_pipeline",
    train_frac: float = 0.8,
    typo: bool = False,
    output_folder: Text = "results/rasa",
):
    set_log_level(30)
    print (
        "Evaluating pipeline '{}' on dataset '{}' (typo: {}, train_frac: {}).".format(
            pipeline_name, data_path, typo, train_frac
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


if __name__ == "__main__":
    output_folder = "results/rasa"

    data_sets = [
        "data/AddToPlaylist",
        "data/BookRestaurant",
        "data/GetWeather",
        "data/RateBook",
        "data/SearchCreativeWork",
        "data/SearchScreeningEvent",
        "data/BTC",
        "data/re3d",
        "data/WNUT17",
        "data/Ritter",
    ]

    pipelines = [
        # (DEFAULT_PIPELINE, "default_pipeline"),
        # (SPACY_PIPELINE, "spacy_pipeline"),
        (SPACY_NER_PIPELINE, "spacy_ner_pipeline"),
        # (FLAIR_PIPELINE, "flair_pipeline")
    ]

    for typo in [False]:
        for train_frac in [0.8]:
            for pipeline, pipeline_name in pipelines:
                for data_set in data_sets:
                    try:
                        run(
                            data_set,
                            pipeline=pipeline,
                            pipeline_name=pipeline_name,
                            typo=typo,
                            train_frac=train_frac,
                            output_folder=output_folder,
                            runs=5,
                        )
                    except Exception as e:
                        print ("#" * 100)
                        traceback.print_exc()
                        print ("#" * 100)
