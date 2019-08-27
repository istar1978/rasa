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


DEFAULT_PIPELINE = [{"name": "WhitespaceTokenizer"}, {"name": "CRFEntityExtractor"}]

SPACY_PIPELINE = [
    {"name": "SpacyNLP"},
    {"name": "SpacyTokenizer"},
    {"name": "SpacyFeaturizer"},
    {"name": "CRFEntityExtractor"},
]

SPACY_NER_PIPELINE = [
    {"name": "SpacyNLP"},
    {"name": "SpacyTokenizer"},
    {"name": "SpacyEntityExtractor"},
]


def train_model(pipeline: List[Dict], data_train: TrainingData) -> Interpreter:
    config = RasaNLUModelConfig(configuration_values={"pipeline": pipeline})
    trainer = Trainer(config)
    return trainer.train(data_train)


def evaluate_model(interpreter: Interpreter, test_data: TrainingData) -> Dict:
    interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)

    result = {"entity_evaluation": None}

    _, entity_results = get_eval_data(interpreter, test_data)

    extractors = get_entity_extractors(interpreter)

    if entity_results:
        result["entity_evaluation"] = evaluate_entities(
            entity_results, extractors, None
        )

    return result["entity_evaluation"][extractors.pop()]


def run(
    data_path: Text,
    runs: int = 5,
    pipeline: List[Dict] = DEFAULT_PIPELINE,
    pipeline_name: Text = "default_pipeline",
    train_frac: float = 0.8,
):
    print ("Evaluating pipeline '{}' on dataset '{}'.".format(pipeline_name, data_path))

    data_set = os.path.splitext(os.path.basename(data_path))[0]
    report_file, result_file, configuration_file = create_output_files(
        data_set, result_folder="results/rasa/{}/".format(pipeline_name)
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
    data_sets = [
        "data/AddToPlaylist.json",
        "data/BookRestaurant.json",
        "data/GetWeather.json",
        "data/RateBook.json",
        "data/SearchCreativeWork.json",
        "data/SearchScreeningEvent.json",
        "data/BTC.md",
        "data/re3d.md",
        "data/WNUT17.md",
        "data/Ritter.md",
        # typo
        "data/typo_AddToPlaylist.json",
        "data/typo_BookRestaurant.json",
        "data/typo_GetWeather.json",
        "data/typo_RateBook.json",
        "data/typo_SearchCreativeWork.json",
        "data/typo_SearchScreeningEvent.json",
        "data/typo_BTC.md",
        "data/typo_re3d.md",
        "data/typo_WNUT17.md",
        "data/typo_Ritter.md",
    ]

    for data_set in data_sets:
        try:
            run(data_set, pipeline=DEFAULT_PIPELINE, pipeline_name="default_pipeline")
        except Exception as e:
            print (e)

        try:
            run(data_set, pipeline=SPACY_PIPELINE, pipeline_name="spacy_pipeline")
        except Exception as e:
            print (e)

        try:
            run(
                data_set,
                pipeline=SPACY_NER_PIPELINE,
                pipeline_name="spacy_ner_pipeline",
            )
        except Exception as e:
            print (e)
