import logging
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Text

import numpy as np
import csv

import matplotlib
import math

# to enable %matplotlib inline if running in ipynb
from IPython import get_ipython

ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic("matplotlib", "inline")


import matplotlib.pyplot as plt


class Plotter(object):
    """
    Plots training parameters (loss, f-score, and accuracy) and training weights over time.
    Input files are the output files 'loss.tsv' and 'weights.txt' from training either a sequence tagger or text
    classification model.
    """

    @staticmethod
    def _extract_evaluation_data(
        file_name: Text, score: str = "loss", ner_intent: str = "ner"
    ) -> dict:
        training_curves = {
            "train": {"loss": [], "score": []},
            "val": {"loss": [], "score": []},
        }

        with open(file_name, "r") as tsvin:
            tsvin = csv.reader(tsvin, delimiter="\t")

            # determine the column index of loss, f-score and accuracy for train, dev and test split
            row = next(tsvin, None)

            score = score.upper()

            TRAIN_SCORE = (
                row.index(f"{ner_intent.upper()}_{score.upper()}")
                if f"{ner_intent.upper()}_{score.upper()}" in row
                else None
            )
            VAL_SCORE = (
                row.index(f"VAL_{ner_intent.upper()}_{score.upper()}")
                if f"VAL_{ner_intent.upper()}_{score.upper()}" in row
                else None
            )

            # then get all relevant values from the tsv
            for row in tsvin:

                if TRAIN_SCORE is not None:
                    if row[TRAIN_SCORE] != "_":
                        training_curves["train"]["score"].append(
                            float(row[TRAIN_SCORE])
                        )

                if VAL_SCORE is not None:
                    if row[VAL_SCORE] != "_":
                        training_curves["val"]["score"].append(float(row[VAL_SCORE]))

        return training_curves

    def plot_training_curves(
        self,
        file_name: Union[Text],
        output_path: Text,
        plot_values: List[str] = ["loss", "acc"],
    ):
        if type(file_name) is str:
            file_name = Path(file_name)

        fig = plt.figure(figsize=(15, 10))

        for plot_no_1, ner_intent_value in enumerate(["ner", "intent"]):
            for plot_no_2, plot_value in enumerate(plot_values):

                training_curves = self._extract_evaluation_data(
                    file_name, plot_value, ner_intent_value
                )

                plt.subplot(
                    len(plot_values) * 2 + 1,
                    1,
                    plot_no_1 * len(plot_values) + plot_no_2 + 1,
                )
                if training_curves["train"]["score"]:
                    x = np.arange(0, len(training_curves["train"]["score"]))
                    plt.plot(
                        x,
                        training_curves["train"]["score"],
                        label=f"train {ner_intent_value} {plot_value}",
                    )
                if training_curves["val"]["score"]:
                    x = np.arange(0, len(training_curves["val"]["score"]))
                    plt.plot(
                        x,
                        training_curves["val"]["score"],
                        label=f"val {ner_intent_value} {plot_value}",
                    )

                plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
                plt.ylabel(ner_intent_value + " " + plot_value)
                plt.xlabel("epochs")

        # save plots
        plt.tight_layout(pad=1.0)
        plt.savefig(output_path, dpi=300)
        print (
            f"Loss and acc plots are saved in {output_path}"
        )  # to let user know the path of the save plots
        plt.close(fig)
