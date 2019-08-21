import io
import os
import pickle
from typing import Any, Optional, Dict, Text
import numpy as np
import shutil
import time
from tqdm import tqdm

import tensorflow as tf
from tensorflow.contrib import predictor

from rasa_nlu.components import Component
from rasa_nlu.training_data import Message
from rasa.nlu.classifiers.bert.run_classifier import (
    create_tokenizer_from_hub_module,
    get_labels,
    get_train_examples,
    convert_examples_to_features,
    build_input_dataset,
    build_model,
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
        "max_checkpoints_to_keep": 1000,
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
        "load_pruning_masks_from_checkpoint": False,
        "resize_pruned_matrices": False,
        "create_inference_graph": False,
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
            logger.warning("'begin_pruning_epoch' should be < 'epochs'.")
        if self.end_pruning_epoch < self.begin_pruning_epoch:
            logger.warning(
                "'end_pruning_epoch' must be >= 'begin_pruning_epoch', setting it to 'begin_pruning_epoch'"
            )
            self.end_pruning_epoch = self.begin_pruning_epoch

        self.pruning_frequency_steps = config["pruning_frequency_steps"]
        self.load_pruning_masks_from_checkpoint = config[
            "load_pruning_masks_from_checkpoint"
        ]
        self.resize_pruned_matrices = config["resize_pruned_matrices"]

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
        self.max_checkpoints_to_keep = config["max_checkpoints_to_keep"]
        self.warm_start_checkpoint = config["warm_start_checkpoint"]
        self.hat_layer_in_checkpoint = config["hat_layer_in_checkpoint"]
        self.finetune_hat_only = config["finetune_hat_layer_only"]
        self.create_inference_graph = config["create_inference_graph"]

    def _load_params(self) -> None:
        self._load_bert_params(self.component_config)
        self._load_train_params(self.component_config)
        self._load_sparsification_params(self.component_config)

    def __init__(
        self,
        component_config=None,
        session: Optional["tf.Session"] = None,
        label_list: Optional[np.ndarray] = None,
        # predict_fn: Optional["Predictor"] = None,
        use_tflite=False,
        important_tensors=None,
    ) -> None:
        super(BertIntentClassifier, self).__init__(component_config)

        tf.logging.set_verbosity(tf.logging.INFO)

        self.session = session
        self.label_list = label_list
        # self.predict_fn = predict_fn
        self.important_tensors = important_tensors

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

        # self.estimator = None

        self.use_tflite = use_tflite

    def _train_tf_dataset(
        self,
        train_init_op,
        batch_size_in,
        batch_size,
        loss: "tf.Tensor",
        train_op: "tf.Tensor",
        train_accuracy: "tf.Tensor",
        num_train_steps=None,
        train_steps_per_epoch=None,
    ) -> None:
        """Train tf graph"""
        saver = tf.train.Saver(
            max_to_keep=self.max_checkpoints_to_keep, name="save_training_checkpoints"
        )
        init = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()
        )
        self.session.run(init)
        # self.session.run(tf.global_variables_initializer())

        pbar = tqdm(range(max(self.epochs, 1)), desc="Epochs")
        # train_acc = 0
        # last_loss = 0
        train_step = 0
        for ep in pbar:
            self.session.run(train_init_op, feed_dict={batch_size_in: batch_size})
            ep_loss = 0
            batches_per_epoch = 0
            while (self.epochs == 0 and batches_per_epoch < num_train_steps) or (
                self.epochs > 0 and batches_per_epoch < train_steps_per_epoch
            ):
                try:
                    _, batch_loss, batch_acc = self.session.run(
                        (train_op, loss, train_accuracy), feed_dict={}
                    )

                except tf.errors.OutOfRangeError:
                    break

                batches_per_epoch += 1
                ep_loss += batch_loss

                if train_step % 10 == 0:
                    print (
                        "accuracy:{:.3f};loss:{:.4f};train_step:{}".format(
                            batch_acc[0], batch_loss, train_step
                        )
                    )

                if train_step % self.save_checkpoints_steps == 0:
                    save_path = saver.save(
                        sess=self.session,
                        save_path=self.checkpoint_dir + "/model.ckpt",
                        global_step=train_step,
                    )
                    print (
                        "Saved checkpoint for step {}: {}".format(train_step, save_path)
                    )
                train_step += 1

            ep_loss /= batches_per_epoch
            pbar.set_postfix({"loss": "{:.3f}".format(ep_loss)})

        save_path = saver.save(
            sess=self.session,
            save_path=self.checkpoint_dir + "/model.ckpt",
            global_step=train_step,
        )
        print ("Saved checkpoint for step {}: {}".format(train_step, save_path))

    def train(self, training_data, cfg, **kwargs):
        """Train this component."""

        # Clean up checkpoint
        if self.checkpoint_remove_before_training and os.path.exists(
            self.checkpoint_dir
        ):
            shutil.rmtree(self.checkpoint_dir, ignore_errors=True)

        self.label_list = get_labels(training_data)

        train_examples = get_train_examples(training_data.training_examples)
        num_train_steps = int(len(train_examples) / self.batch_size * self.epochs)
        train_steps_per_epoch = int(len(train_examples) / self.batch_size)
        min_steps = 1
        if self.epochs <= 0:
            num_train_steps = min_steps
        print ("RUNNING {} EPOCHS, {} STEPS".format(self.epochs, num_train_steps))
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        tf.logging.info("***** Running training *****")
        tf.logging.info("Num examples = %d", len(train_examples))
        tf.logging.info("Batch size = %d", self.batch_size)
        tf.logging.info("Num steps = %d", num_train_steps)
        tf.logging.info("Num epochs = %d", self.epochs)
        train_features = convert_examples_to_features(
            train_examples, self.label_list, self.max_seq_length, self.tokenizer
        )

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

        begin_pruning_step = train_steps_per_epoch * self.begin_pruning_epoch
        end_pruning_step = max(
            train_steps_per_epoch * self.end_pruning_epoch, begin_pruning_step + 1
        )
        if self.epochs <= 0:
            end_pruning_step = begin_pruning_step + min_steps

        logger.info(
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
                "resize_pruned_matrices": self.resize_pruned_matrices,
                "checkpoint_for_pruning_masks": self.warm_start_checkpoint,
            },
            "finetune_hat_only": self.finetune_hat_only,
        }
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            if not self.create_inference_graph:
                train_dataset = build_input_dataset(
                    features=train_features,
                    seq_length=self.max_seq_length,
                    is_training=True,
                    drop_remainder=True,
                    params=params,
                )
                batch_size_in = tf.placeholder(tf.int64)
                iterator = tf.data.Iterator.from_structure(
                    output_types=train_dataset.output_types,
                    output_shapes=train_dataset.output_shapes,
                    output_classes=train_dataset.output_classes,
                )
                minibatch = iterator.get_next()
                train_init_op = iterator.make_initializer(train_dataset)
                train_op, loss, input_tensors, log_probs, predictions, train_acc = build_model(
                    features=minibatch,
                    mode="train",
                    params=params,
                    bert_tfhub_module_handle=self.bert_tfhub_module_handle,
                    num_labels=len(self.label_list),
                    learning_rate=self.learning_rate,
                    num_train_steps=num_train_steps,
                    num_warmup_steps=num_warmup_steps,
                    bert_config=bert_config,
                )

                writer = tf.summary.FileWriter(
                    logdir="tfgraph-bert-train", graph=self.graph
                )
                writer.flush()

                trainable_vars = self.graph.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES
                )
                if self.hat_layer_in_checkpoint:
                    assignment_map = {
                        v.name.split(":")[0]: v.name.split(":")[0]
                        for v in trainable_vars
                    }
                else:
                    assignment_map = {
                        v.name.split(":")[0]: v.name.split(":")[0]
                        for v in trainable_vars
                        if v.name.startswith("bert/")
                    }
                tf.train.init_from_checkpoint(
                    self.warm_start_checkpoint, assignment_map=assignment_map
                )
                self.session = tf.Session()
                self._train_tf_dataset(
                    train_init_op=train_init_op,
                    batch_size_in=batch_size_in,
                    batch_size=self.batch_size,
                    loss=loss,
                    train_op=train_op,
                    train_accuracy=train_acc,
                    num_train_steps=num_train_steps,  # (None if self.epochs > 0 else num_train_steps),
                    train_steps_per_epoch=train_steps_per_epoch,
                )

                self.input_tensors = input_tensors
                self.input_tensors["probabilities"] = log_probs
                self.input_tensors["predictions"] = predictions

                # exit(0)
            else:
                train_dataset = build_input_dataset(
                    features=train_features,
                    seq_length=self.max_seq_length,
                    is_training=False,
                    drop_remainder=True,
                    params=params,
                )
                iterator = tf.data.Iterator.from_structure(
                    output_types=train_dataset.output_types,
                    output_shapes=train_dataset.output_shapes,
                    output_classes=train_dataset.output_classes,
                )
                minibatch = iterator.get_next()
                self.input_tensors = {}
                for input_name, input_tensor in minibatch.items():
                    placeholder = tf.placeholder(
                        input_tensor.dtype, input_tensor.shape, name=input_name
                    )
                    self.input_tensors[input_name] = placeholder

                predictions, log_probs, input_tensors = build_model(
                    features=self.input_tensors,
                    mode="predict",
                    params=params,
                    bert_tfhub_module_handle=self.bert_tfhub_module_handle,
                    num_labels=len(self.label_list),
                    learning_rate=self.learning_rate,
                    num_train_steps=num_train_steps,
                    num_warmup_steps=num_warmup_steps,
                    bert_config=bert_config,
                )

                writer = tf.summary.FileWriter(
                    logdir="tfgraph-bert-predict", graph=self.graph
                )
                writer.flush()
                # exit(0)

                self.input_tensors["probabilities"] = log_probs
                self.input_tensors["predictions"] = predictions

                trainable_vars = self.graph.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES
                )

                assignment_map = {
                    v.name.split(":")[0]: v.name.split(":")[0] for v in trainable_vars
                }
                tf.train.init_from_checkpoint(
                    self.warm_start_checkpoint, assignment_map=assignment_map
                )

                self.session = tf.Session()
                self.session.run(tf.global_variables_initializer())

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

        # size compression: from 408mb to around 110mb in all cases where optimisation is used, without opt it's ~437mb
        # full (estimator):        116s (0.118s/message). micro f1: 0.884, macro f1: 0.921
        # tflite, no optims:       597s (0.605s/message). micro f1: 0.884, macro f1: 0.921
        # tflite, opt_for_size:    1961s (1.99s/message). micro f1: 0.880, macro f1: 0.918
        # tflite, opt_for_latency: 2040s (2.07s/message). micro f1: 0.880, macro f1: 0.918
        # weight pruned:           241s (0.244s/message).

        # full (bare)              141s (0.143s/message), 406mb.
        # neuron pruned (0% avg)   156s (0.158s/message), 406mb. (measured the overhead created by scattering things)
        # neuron pruned (50% avg)  142s (0.144s/message), 271mb. (k:.9 q:.9 v:.45 ao:.25 i:.45 o:.2 p:.3)
        # neuron pruned (50% avg)  122s (0.124s/message), ???mb. (cross-pruning of layer output weight matrix)
        # neuron pruned (50% fix)  134s (0.136s/message), 247mb.
        # neuron pruned (50% fix)  112s (0.113s/message), 220mb. (cross-pruning of layer output weight matrix)
        # neuron pruned (56.5% avg)105s (0.106s/message), 205mb. (pruning intermed. more: k:.9 q:.9 v:.45 ao:.25 i:.6 o:.2 p:.3)
        # neuron pruned (55.9% avg)111s (0.112s/message), 210mb. (everything except k & q pruned *1.2)
        # neuron pruned (58.5% avg)103s (0.105s/message), 199mb. (everything except k & q pruned *1.2 and i:.6)

        start = time.time()
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
        else:
            probs_op = self.important_tensors["probabilities"]

            log_probs = self.session.run(
                probs_op,
                feed_dict={
                    self.important_tensors["input_ids"]: np.array(
                        example.input_ids
                    ).reshape(-1, self.max_seq_length),
                    self.important_tensors["input_mask"]: np.array(
                        example.input_mask
                    ).reshape(-1, self.max_seq_length),
                    # "label_ids": np.array(example.label_id).reshape(-1),
                    self.important_tensors["segment_ids"]: np.array(
                        example.segment_ids
                    ).reshape(-1, self.max_seq_length),
                },
            )
            probabilities = list(np.exp(log_probs)[0])
        print ("inference time: {:.3f}".format(time.time() - start))

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

        if self.session is None:
            return {"file": None}

        checkpoint = os.path.join(model_dir, file_name + ".ckpt")

        try:
            os.makedirs(os.path.dirname(checkpoint))
        except OSError as e:
            # be happy if someone already created the path
            import errno

            if e.errno != errno.EEXIST:
                raise

        with io.open(os.path.join(model_dir, self.name + "_label_list.pkl"), "wb") as f:
            pickle.dump(self.label_list, f)

        with self.graph.as_default():
            for in_name, in_tensor in self.input_tensors.items():
                self.graph.clear_collection(in_name)
                self.graph.add_to_collection(in_name, in_tensor)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: "Metadata" = None,
        cached_component: Optional["BertIntentClassifier"] = None,
    ) -> "BertIntentClassifier":
        if not (model_dir and meta.get("file")):
            logger.warning(
                "Failed to load nlu model. Maybe path {} "
                "doesn't exist"
                "".format(os.path.abspath(model_dir))
            )
            return cls(component_config=meta)

        with io.open(os.path.join(model_dir, cls.name + "_label_list.pkl"), "rb") as f:
            label_list = pickle.load(f)

        file_name = meta.get("file")
        checkpoint = os.path.join(model_dir, file_name + ".ckpt")

        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            saver = tf.train.import_meta_graph(checkpoint + ".meta")

            input_tensors = {}
            for in_tensor_name in [
                "input_ids",
                "input_mask",
                "segment_ids",
                "label_ids",
                "probabilities",
                "predictions",
            ]:
                in_tensor = tf.get_collection(in_tensor_name)[0]
                input_tensors[in_tensor_name] = in_tensor

            saver.restore(sess, checkpoint)

        writer = tf.summary.FileWriter(logdir="tfgraph-bert-loaded", graph=graph)
        writer.flush()
        return cls(
            component_config=meta,
            session=sess,
            label_list=label_list,
            use_tflite=False,
            important_tensors=input_tensors,
        )
