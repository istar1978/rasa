import io
import os
import pickle
from typing import Any, Optional, Dict, Text
import numpy as np
import shutil
import time

import tensorflow as tf
from tensorflow.contrib import predictor

from rasa_nlu.components import Component
from rasa_nlu.training_data import Message
from rasa.nlu.classifiers.bert.run_classifier import (
    create_tokenizer_from_hub_module,
    get_labels,
    get_train_examples,
    convert_examples_to_features,
    model_fn_builder,
    input_fn_builder,
    serving_input_fn_builder,
    get_test_examples,
)
from rasa.nlu.classifiers.bert.tokenization import FullTokenizer
from rasa.nlu.classifiers.bert.modeling import BertConfig
import logging

logger = logging.getLogger(__name__)


class BertIntentClassifier(Component):
    """
    Intent classifier using BERT.
    """

    name = "BertIntentClassifier"

    provides = ["intent", "intent_ranking"]

    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12

    defaults = {
        "batch_size": 64,
        "epochs": 2,
        "learning_rate": 2e-5,
        "max_seq_length": 128,
        "warmup_proportion": 0.1,
        "save_checkpoints_steps": 1000,
        "save_summary_steps": 500,
        "bert_tfhub_module_handle": "https://tfhub.dev/google/bert_uncased_L-{}_H-{}_A-{}/1".format(
            num_hidden_layers, hidden_size, num_attention_heads
        ),
        "pretrained_model_dir": None,
        "checkpoint_dir": "./tmp/bert",
        "checkpoint_remove_before_training": True,
        "warm_start_checkpoint": None,
        "hat_layer_in_checkpoint": False,
        "use_tflite": False,
        "sparsity_technique": None,
        "target_sparsity": 0.5,
        "begin_pruning_epoch": 0,
        "end_pruning_epoch": 2,
        "pruning_frequency_steps": 1,
        "finetune_hat_layer_only": False,
    }

    def _load_bert_params(self, config: Dict[Text, Any]) -> None:
        self.bert_tfhub_module_handle = config["bert_tfhub_module_handle"]
        self.pretrained_model_dir = config["pretrained_model_dir"]

        if self.pretrained_model_dir:
            dir_files = os.listdir(self.pretrained_model_dir)
            if all(file not in dir_files for file in ("bert_config.json", "vocab.txt")):
                logger.warning(
                    "Pretrained model dir configured as '{}' "
                    "does not contain 'bert_config.json', "
                    "'vocab.txt' and a model checkpoint.  "
                    "No pretrained model was loaded."
                )
                logger.warning("Using TF Hub module instead.")
                self.pretrained_model_dir = None
            else:
                logger.info(
                    "Loading pretrained model from {}".format(self.pretrained_model_dir)
                )

    def _load_sparsification_params(self, config: Dict[Text, Any]) -> None:
        self.sparsity_technique = config["sparsity_technique"]
        self.target_sparsity = config["target_sparsity"]
        self.begin_pruning_epoch = config["begin_pruning_epoch"]
        self.end_pruning_epoch = config["end_pruning_epoch"]
        if self.begin_pruning_epoch > self.epochs:
            logger.warning(
                "'begin_pruning_epoch' must be < 'epochs', setting it to epochs-1."
            )
            self.begin_pruning_epoch = max(0, self.epochs - 1)
        if (
            self.end_pruning_epoch > self.epochs
            or self.end_pruning_epoch <= self.begin_pruning_epoch
        ):
            logger.warning(
                "'end_pruning_epoch' must be <= 'epochs' and > 'begin_pruning_epoch', setting it to highest possible correct value"
            )
            self.end_pruning_epoch = min(
                self.epochs, max(self.end_pruning_epoch, self.begin_pruning_epoch)
            )
        self.pruning_frequency_steps = config["pruning_frequency_steps"]

    def _load_train_params(self, config: Dict[Text, Any]) -> None:
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.learning_rate = config["learning_rate"]
        self.max_seq_length = config["max_seq_length"]
        self.warmup_proportion = config["warmup_proportion"]
        self.save_checkpoints_steps = config["save_checkpoints_steps"]
        self.save_summary_steps = config["save_summary_steps"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.checkpoint_remove_before_training = config[
            "checkpoint_remove_before_training"
        ]
        self.warm_start_checkpoint = config["warm_start_checkpoint"]
        self.hat_layer_in_checkpoint = config["hat_layer_in_checkpoint"]
        self.finetune_hat_only = config["finetune_hat_layer_only"]

    def _load_params(self) -> None:
        self._load_bert_params(self.component_config)
        self._load_train_params(self.component_config)
        self._load_sparsification_params(self.component_config)

    def __init__(
        self,
        component_config=None,
        session: Optional["tf.Session"] = None,
        label_list: Optional[np.ndarray] = None,
        predict_fn: Optional["Predictor"] = None,
        use_tflite=False,
    ) -> None:
        super(BertIntentClassifier, self).__init__(component_config)

        tf.logging.set_verbosity(tf.logging.INFO)

        self.session = session
        self.label_list = label_list
        self.predict_fn = predict_fn

        self._load_params()

        if self.pretrained_model_dir:
            vocab_file = os.path.join(self.pretrained_model_dir, "vocab.txt")
            do_lower_case = os.path.basename(self.pretrained_model_dir).startswith(
                "uncased"
            )

            self.tokenizer = FullTokenizer(
                vocab_file=vocab_file, do_lower_case=do_lower_case
            )
        else:
            self.tokenizer = create_tokenizer_from_hub_module(
                self.bert_tfhub_module_handle
            )

        self.estimator = None

        self.use_tflite = use_tflite

    def train(self, training_data, cfg, **kwargs):
        """Train this component."""

        # Clean up checkpoint
        if self.checkpoint_remove_before_training and os.path.exists(
            self.checkpoint_dir
        ):
            shutil.rmtree(self.checkpoint_dir, ignore_errors=True)

        self.label_list = get_labels(training_data)

        run_config = tf.estimator.RunConfig(
            model_dir=self.checkpoint_dir,
            save_summary_steps=self.save_summary_steps,
            save_checkpoints_steps=self.save_checkpoints_steps,
        )

        train_examples = get_train_examples(training_data.training_examples)
        num_train_steps = int(len(train_examples) / self.batch_size * self.epochs)
        train_steps_per_epoch = int(len(train_examples) / self.batch_size)
        if self.epochs <= 0:
            num_train_steps = 20
        print ("RUNNING {} EPOCHS, {} STEPS".format(self.epochs, num_train_steps))
        # exit(0)
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        tf.logging.info("***** Running training *****")
        tf.logging.info("Num examples = %d", len(train_examples))
        tf.logging.info("Batch size = %d", self.batch_size)
        tf.logging.info("Num steps = %d", num_train_steps)
        tf.logging.info("Num epochs = %d", self.epochs)
        train_features = convert_examples_to_features(
            train_examples, self.label_list, self.max_seq_length, self.tokenizer
        )
        # exit(0)

        """
        # creating small representative dataset for full post-training quantisation with TFLite
        calibration_data = []
        calibration_sample_indices = np.random.choice(range(len(train_features)), size=100, replace=False)
        for idx in calibration_sample_indices:
            f = train_features[idx]
            datapoint = []
            datapoint.append(np.array(f.input_ids).reshape(-1, self.max_seq_length).astype(np.int32))
            datapoint.append(np.array(f.input_mask).reshape(-1, self.max_seq_length).astype(np.int32))
            datapoint.append(np.array(f.label_id).reshape(-1).astype(np.int32))
            datapoint.append(np.array(f.segment_ids).reshape(-1, self.max_seq_length).astype(np.int32))
            calibration_data.append(datapoint)

        with open('data/calibration_data.pickle', 'wb') as handle:
            pickle.dump(calibration_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        """

        if self.pretrained_model_dir:
            bert_config = BertConfig.from_json_file(
                os.path.join(self.pretrained_model_dir, "bert_config.json")
            )
        else:
            bert_config = None

        model_fn = model_fn_builder(
            bert_tfhub_module_handle=self.bert_tfhub_module_handle,
            num_labels=len(self.label_list),
            learning_rate=self.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            bert_config=bert_config,
        )

        train_input_fn = input_fn_builder(
            features=train_features,
            seq_length=self.max_seq_length,
            is_training=True,
            drop_remainder=True,
        )

        begin_pruning_step = train_steps_per_epoch * self.begin_pruning_epoch
        end_pruning_step = max(
            train_steps_per_epoch * self.end_pruning_epoch, begin_pruning_step + 1
        )
        print (
            "Begin pruning: {}, end: {}".format(begin_pruning_step, end_pruning_step)
        )

        params = {
            "batch_size": self.batch_size,
            "sparsity_technique": self.sparsity_technique,
            "sparsification_params": {
                "begin_pruning_step": begin_pruning_step,
                "end_pruning_step": end_pruning_step,
                "pruning_frequency": self.pruning_frequency_steps,
                "target_sparsity": self.target_sparsity,
            },
            "finetune_hat_only": self.finetune_hat_only,
        }

        warm_start_settings = None
        if self.warm_start_checkpoint:
            if self.hat_layer_in_checkpoint:
                warm_start_settings = tf.estimator.WarmStartSettings(
                    self.warm_start_checkpoint
                )
            else:
                warm_start_settings = tf.estimator.WarmStartSettings(
                    self.warm_start_checkpoint, vars_to_warm_start="bert.*"
                )

        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params=params,
            model_dir=self.checkpoint_dir,
            warm_start_from=warm_start_settings,
        )

        # Start training
        self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        serving_input_receiver_fn = serving_input_fn_builder(self.max_seq_length)

        # Create predictor incase running evaluation
        self.predict_fn = predictor.from_estimator(
            self.estimator, serving_input_receiver_fn
        )

        g = self.predict_fn.graph
        writer = tf.summary.FileWriter(logdir="tfgraph-bert-pruned", graph=g)
        writer.flush()

        """
        Difference: Tom's weights have module/ wrapped around bert/ and these vars are added:
        cls/predictions/output_bias
        cls/predictions/transform/dense/bias
        cls/predictions/transform/dense/kernel
        cls/predictions/transform/LayerNorm/beta
        cls/predictions/transform/LayerNorm/gamma
        global_step

        Also, as a result of weight pruning, the naming changes like so: kernel->weights and bias->biases
        Additionally, weight pruning creates weight masks, but no need to worry about those.
        """

    def process(self, message: Message, **kwargs: Any) -> None:
        """Return the most likely intent and its similarity to the input"""

        # Classifier needs this to be non empty, so we set to first label.
        message.data["intent"] = self.label_list[0]
        predict_examples = get_test_examples([message])
        predict_features = convert_examples_to_features(
            predict_examples, self.label_list, self.max_seq_length, self.tokenizer
        )

        # Get first index since we are only classifying text blob at a time.
        example = predict_features[0]

        # start = time.time()
        if self.use_tflite:
            input_ids = (
                np.array(example.input_ids)
                .reshape(-1, self.max_seq_length)
                .astype(np.int32)
            )
            self.interpreter.set_tensor(self.in_indices["input_ids"], input_ids)

            input_mask = (
                np.array(example.input_mask)
                .reshape(-1, self.max_seq_length)
                .astype(np.int32)
            )
            self.interpreter.set_tensor(self.in_indices["input_mask"], input_mask)

            label_ids = np.array(example.label_id).reshape(-1).astype(np.int32)
            self.interpreter.set_tensor(self.in_indices["label_ids"], label_ids)

            segment_ids = (
                np.array(example.segment_ids)
                .reshape(-1, self.max_seq_length)
                .astype(np.int32)
            )
            self.interpreter.set_tensor(self.in_indices["segment_ids"], segment_ids)

            self.interpreter.invoke()
            result = self.interpreter.get_tensor(self.out_index)
            probabilities = list(np.exp(result[0]))
            # size compression: from 408mb to around 110mb in all cases where optimisation is used, without opt it's ~437mb
            # full:               116s (0.118s/message). micro f1: 0.884, macro f1: 0.921
            # tflite, no optims:  597s (0.605s/message). micro f1: 0.884, macro f1: 0.921
            # tflite, size:       1961s (1.99s/message). micro f1: 0.880, macro f1: 0.918
            # tflite, latency:    2040s (2.07s/message). micro f1: 0.880, macro f1: 0.918
            # weight pruned:      241s (0.244s/message).
        else:
            result = self.predict_fn(
                {
                    "input_ids": np.array(example.input_ids).reshape(
                        -1, self.max_seq_length
                    ),
                    "input_mask": np.array(example.input_mask).reshape(
                        -1, self.max_seq_length
                    ),
                    "label_ids": np.array(example.label_id).reshape(-1),
                    "segment_ids": np.array(example.segment_ids).reshape(
                        -1, self.max_seq_length
                    ),
                }
            )
            probabilities = list(np.exp(result["probabilities"])[0])
        # print ("inference time: {:.3f}".format(time.time() - start))

        index = np.argmax(probabilities)
        label = self.label_list[index]
        score = float(probabilities[index])

        intent = {"name": label, "confidence": score}
        intent_ranking = sorted(
            [
                {"name": self.label_list[i], "confidence": float(score)}
                for i, score in enumerate(probabilities)
            ],
            key=lambda k: k["confidence"],
            reverse=True,
        )

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

        try:
            os.makedirs(model_dir)
        except OSError as e:
            # Be happy if someone already created the path
            import errno

            if e.errno != errno.EEXIST:
                raise
        model_path = self.estimator.export_saved_model(
            model_dir, serving_input_fn_builder(self.max_seq_length)
        )

        with io.open(os.path.join(model_dir, self.name + "_label_list.pkl"), "wb") as f:
            pickle.dump(self.label_list, f)

        return {"model_path": model_path.decode("UTF-8")}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: "Metadata" = None,
        cached_component: Optional["EmbeddingIntentClassifier"] = None,
        **kwargs: Any
    ) -> "BertIntentClassifier":

        if model_dir and meta.get("model_path"):

            with io.open(
                os.path.join(model_dir, cls.name + "_label_list.pkl"), "rb"
            ) as f:
                label_list = pickle.load(f)

            model_path = os.path.join(
                model_dir, [p for p in os.walk(model_dir)][0][1][0]
            )

            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session()
                predict_fn = predictor.from_saved_model(model_path)

                writer = tf.summary.FileWriter(
                    logdir="tfgraph-bert-pruned-test", graph=predict_fn.graph
                )
                writer.flush()

                if meta["use_tflite"]:
                    tflite_model_file = "tflite/converted_model_bert.tflite"
                    in_tensor_names = [
                        "input_ids",
                        "input_mask",
                        "label_ids",
                        "segment_ids",
                    ]

                    in_tensors = [
                        predict_fn.feed_tensors[name] for name in in_tensor_names
                    ]
                    out_tensors = [t for _, t in predict_fn.fetch_tensors.items()]
                    converter = tf.lite.TFLiteConverter.from_session(
                        predict_fn.session, in_tensors, out_tensors
                    )

                    # """
                    converter.optimizations = [
                        tf.lite.Optimize.DEFAULT  # 4s (~219it/s) (4.604s invoking time)
                        # tf.lite.Optimize.OPTIMIZE_FOR_SIZE # 4s (~220it/s) (4.707s invoking time)
                        # tf.lite.Optimize.OPTIMIZE_FOR_LATENCY # 4s (~215it/s) (4.630s invoking time)
                    ]
                    # """

                    """
                    # representative dataset: generator, each element is a list of input items, 
                    # items are passed to in_tensors in the order established when creating the converter
                    # https://github.com/tensorflow/tensorflow/blob/d883916ee45f3af81e81eefb7e8495d9fab6d231/tensorflow/lite/python/optimize/calibration_wrapper.cc#L110

                    with open('data/calibration_data.pickle', 'rb') as handle:
                        calibration_data = pickle.load(handle)

                    def representative_dataset_gen():
                        for d in calibration_data[:5]:
                            yield d
                    converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
                    """

                    # https://github.com/tensorflow/tensorflow/blob/61128913681a016033143fbe9b60140d983b3c98/tensorflow/lite/tools/optimize/quantize_model.cc
                    tflite_model = converter.convert()
                    open(tflite_model_file, "wb").write(tflite_model)
                    # """

                    obj = cls(
                        component_config=meta,
                        session=sess,
                        label_list=label_list,
                        predict_fn=predict_fn,
                        use_tflite=True,
                    )

                    obj.interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
                    obj.interpreter.allocate_tensors()

                    obj.in_indices = {}
                    for i, in_name in enumerate(in_tensor_names):
                        # print(in_name, obj.interpreter.get_input_details()[i]["shape"])
                        obj.in_indices[in_name] = obj.interpreter.get_input_details()[
                            i
                        ]["index"]

                    obj.out_index = obj.interpreter.get_output_details()[0]["index"]
                else:
                    obj = cls(
                        component_config=meta,
                        session=sess,
                        label_list=label_list,
                        predict_fn=predict_fn,
                        use_tflite=False,
                    )

                return obj
        else:
            logger.warning(
                "Failed to load nlu model. Maybe path {} "
                "doesn't exist"
                "".format(os.path.abspath(model_dir))
            )
            return cls(component_config=meta)
