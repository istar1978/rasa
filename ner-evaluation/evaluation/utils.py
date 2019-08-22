import os
from typing import Text, Tuple, Optional

from nlu import load_data
from nlu.training_data import TrainingData


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


def create_output_files(
    data_set: Text, result_folder: Text = "results"
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


def write_results(
    result_file: Text, accuracy: float, f1_score: float, precision: float
):
    f = open(result_file, "a")
    f.write("RESULTS\n")
    f.write("accuracy: {}\n".format(str(accuracy)))
    f.write("f1_score: {}\n".format(str(f1_score)))
    f.write("precision: {}\n".format(str(precision)))
    f.close()


def add_to_report(report_file: Text, run: int, report: Text):
    f = open(report_file, "a")
    f.write("#" * 100)
    f.write("\n")
    f.write("RUN {}\n\n".format(run))
    f.write(report)
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
