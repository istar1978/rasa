import json
import os
from typing import Text, Tuple, Optional, Dict

from rasa.nlu.training_data.loading import load_files
from rasa.nlu.training_data import TrainingData
from rasa.utils.io import list_files


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
) -> Tuple[Text, Text, Text]:
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
    pipeline: Optional[Text] = None,
):
    f = open(configuration_file, "w")
    f.write("CONFIG\n")
    f.write("pipeline: {}\n".format(pipeline))
    f.write("runs: {}\n".format(runs))
    f.write("training data: {}\n".format(train_frac))
    f.close()
