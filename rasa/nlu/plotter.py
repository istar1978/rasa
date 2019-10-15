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
    def _extract_evaluation_data(file_name: Text, score: str = "loss") -> dict:
        training_curves = {
            "ner": {"loss": [], "score": []},
            "intent": {"loss": [], "score": []},
        }

        with open(file_name, "r") as tsvin:
            tsvin = csv.reader(tsvin, delimiter="\t")

            # determine the column index of loss, f-score and accuracy for train, dev and test split
            row = next(tsvin, None)

            score = score.upper()

            NER_SCORE = row.index(f"NER_{score}") if f"NER_{score}" in row else None
            INTENT_SCORE = (
                row.index(f"INTENT_{score}") if f"INTENT_{score}" in row else None
            )

            # then get all relevant values from the tsv
            for row in tsvin:

                if NER_SCORE is not None:
                    if row[NER_SCORE] != "_":
                        training_curves["ner"]["score"].append(float(row[NER_SCORE]))

                if INTENT_SCORE is not None:
                    if row[INTENT_SCORE] != "_":
                        training_curves["intent"]["score"].append(
                            float(row[INTENT_SCORE])
                        )

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

        for plot_no, plot_value in enumerate(plot_values):

            training_curves = self._extract_evaluation_data(file_name, plot_value)

            plt.subplot(len(plot_values), 1, plot_no + 1)
            if training_curves["ner"]["score"]:
                x = np.arange(0, len(training_curves["ner"]["score"]))
                plt.plot(x, training_curves["ner"]["score"], label=f"ner {plot_value}")
            if training_curves["intent"]["score"]:
                x = np.arange(0, len(training_curves["intent"]["score"]))
                plt.plot(
                    x, training_curves["intent"]["score"], label=f"intent {plot_value}"
                )
            plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
            plt.ylabel(plot_value)
            plt.xlabel("epochs")

        # save plots
        plt.tight_layout(pad=1.0)
        plt.savefig(output_path, dpi=300)
        print (
            f"Loss and acc plots are saved in {output_path}"
        )  # to let user know the path of the save plots
        plt.close(fig)
