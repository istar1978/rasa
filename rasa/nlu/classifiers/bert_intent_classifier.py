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
from rasa.nlu.classifiers.compression_utils import resize_bert_weights, sparsity_report
import logging

logger = logging.getLogger(__name__)


class BertIntentClassifier(Component):
    """
    Intent classifier using BERT.
    """

    name = "BertIntentClassifier"

    provides = ["intent", "intent_ranking"]

    defaults = {
        "batch_size": 64,  # note that the limit seems to be 16 or 32 on a 12GB GPU if pruning is used
        "epochs": 2,
        "learning_rate": 2e-5,
        "max_seq_length": 128,
        "warmup_proportion": 0.1,
        "save_checkpoints": False,
        "checkpoint_remove_before_training": True,
        "save_checkpoints_steps": 1000,
        "max_checkpoints_to_keep": 1000,
        "save_summary_steps": 500,
        "checkpoint_dir": "./tmp/bert",  # directory to use for saving training checkpoints
        # Directory for saving the temporary checkpoint when creating inference graph after training.
        # Important especially for neuron pruning.
        "tmp_ckpt_name": "tmp/bert-np-resized/model.ckpt",
        "pretrained_model_dir": None,  # in reality, this is only used to get the Bert config
        # this may not work, use loading weights from checkpoint instead
        "bert_tfhub_module_handle": "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        "warm_start_checkpoint": None,
        # whether the added classifier layer is found in the checkpoint or not
        # (e.g. it is not in Google's pre-trained checkpoints because it is only added during fine-tuning).
        "hat_layer_in_checkpoint": False,
        "do_training": True,  # whether to actually train the weights or not
        # checkpoint to use in train() instead of actually training the model when do_training=False (useful
        # for creating the inference graph from existing weights, e.g. to evaluate the performance of a saved checkpoint)
        "no_training_checkpoint": None,
        "tflite_quantise": False,  # DOES NOT WORK AT THE MOMENT
        "sparsity_technique": None,  # can be 'neuron_pruning' or 'weight_pruning'
        "sparsity_function_exponent": 1,  # used in tf.contrib.model_pruning to anneal the pruning speed
        "target_sparsity": 0.5,  # overall target sparsity (expressed as fraction of weights or neurons removed)
        # alternatively, provide a dict like this: {"k":0.1, "q":0.2,"v":0.3, "ao":0.4, "i":0.5, "o":0.6, "p":0.7}
        "component_target_sparsities": None,
        "begin_pruning_epoch": 0,
        "end_pruning_epoch": 2,
        "pruning_frequency_steps": 1,  # how many minibatches to process between two consecutive pruning steps
        "finetune_hat_layer_only": False,  # whether to freeze all layers except the added fine-tuning layer and train only that one
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
        self.component_target_sparsities = config["component_target_sparsities"]
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
        self.sparsity_function_exponent = config["sparsity_function_exponent"]

    def _load_train_params(self, config: Dict[Text, Any]) -> None:
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.learning_rate = config["learning_rate"]
        self.max_seq_length = config["max_seq_length"]
        self.warmup_proportion = config["warmup_proportion"]
        self.save_checkpoints_steps = config["save_checkpoints_steps"]
        self.save_summary_steps = config["save_summary_steps"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.save_checkpoints = config["save_checkpoints"]
        self.checkpoint_remove_before_training = config[
            "checkpoint_remove_before_training"
        ]
        self.max_checkpoints_to_keep = config["max_checkpoints_to_keep"]
        self.warm_start_checkpoint = config["warm_start_checkpoint"]
        self.tmp_ckpt_name = config["tmp_ckpt_name"]
        self.hat_layer_in_checkpoint = config["hat_layer_in_checkpoint"]
        self.finetune_hat_only = config["finetune_hat_layer_only"]

        self.do_training = config["do_training"]
        self.no_training_checkpoint = config["no_training_checkpoint"]

    def _load_params(self) -> None:
        self._load_bert_params(self.component_config)
        self._load_train_params(self.component_config)
        self._load_sparsification_params(self.component_config)

    def __init__(
        self,
        component_config=None,
        session: Optional["tf.Session"] = None,
        label_list: Optional[np.ndarray] = None,
        tflite_quantise=False,
        important_tensors=None,
    ) -> None:
        super(BertIntentClassifier, self).__init__(component_config)

        tf.logging.set_verbosity(tf.logging.INFO)

        self.session = session
        self.label_list = label_list
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

        self.tflite_quantise = tflite_quantise

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

        pbar = tqdm(range(max(self.epochs, 1)), desc="Epochs")
        train_step = 0
        for ep in pbar:
            self.session.run(train_init_op, feed_dict={batch_size_in: batch_size})
            ep_loss = 0
            batches_per_epoch = 0
            while (self.epochs == 0 and batches_per_epoch < num_train_steps) or (
                self.epochs > 0
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
                    tf.logging.info(
                        "accuracy:{:.3f};loss:{:.4f};train_step:{}".format(
                            batch_acc[0], batch_loss, train_step
                        )
                    )

                if (
                    self.save_checkpoints
                    and train_step % self.save_checkpoints_steps == 0
                ):
                    save_path = saver.save(
                        sess=self.session,
                        save_path=self.checkpoint_dir + "/model.ckpt",
                        global_step=train_step,
                    )
                    tf.logging.info(
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
        tf.logging.info(
            "Saved checkpoint for step {}: {}".format(train_step, save_path)
        )

    def train(self, training_data, cfg, **kwargs):
        """Train this component."""

        # Clean up checkpoint
        if self.checkpoint_remove_before_training and os.path.exists(
            self.checkpoint_dir
        ):
            shutil.rmtree(self.checkpoint_dir, ignore_errors=True)

        self.label_list = get_labels(training_data)

        train_examples = get_train_examples(training_data.training_examples)
        num_train_steps = int(len(train_examples) / self.batch_size) * self.epochs
        train_steps_per_epoch = int(len(train_examples) / self.batch_size)
        min_steps = 2
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
        ## Creating small representative dataset for full post-training quantisation with TFLite
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

        params = {
            "batch_size": self.batch_size,
            "sparsity_technique": self.sparsity_technique,
            "sparsification_params": {
                "begin_pruning_step": begin_pruning_step,
                "end_pruning_step": end_pruning_step,
                "pruning_frequency": self.pruning_frequency_steps,
                "target_sparsity": self.target_sparsity,
                "resize_pruned_matrices": False,
                "checkpoint_for_pruning_masks": self.warm_start_checkpoint,
                "sparsity_function_exponent": self.sparsity_function_exponent,
                "component_target_sparsities": self.component_target_sparsities,
            },
            "finetune_hat_only": self.finetune_hat_only,
        }

        # training graph
        if self.do_training:
            # do normal training
            self.graph = tf.Graph()
            with self.graph.as_default() as g:
                with tf.variable_scope("input_feeding", reuse=tf.AUTO_REUSE):
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
                    num_train_steps=num_train_steps,
                    train_steps_per_epoch=train_steps_per_epoch,
                )

                self.input_tensors = input_tensors
                self.input_tensors["probabilities"] = log_probs
                self.input_tensors["predictions"] = predictions
        else:
            # take trained weights for the entire model from a checkpoint instead of training
            self.graph = tf.Graph()
            with self.graph.as_default() as g:
                self.session = tf.Session()
                saver = tf.train.import_meta_graph(
                    self.no_training_checkpoint + ".meta"
                )
                saver.restore(self.session, self.no_training_checkpoint)

        # This just prints out the sparsity (element- and column-wise) of each pruning mask
        if self.sparsity_technique is not None:
            sparsity_report(self.graph, self.session)

        # In case of neuron pruning resize the weights before saving all variables
        # for the inference graph, otherwise simply save everything.
        if self.sparsity_technique == "neuron_pruning":
            resized_weights_ckpt = resize_bert_weights(
                self.graph, self.session, tmp_ckpt_name=self.tmp_ckpt_name
            )
            self.warm_start_checkpoint = resized_weights_ckpt
            params["sparsification_params"][
                "checkpoint_for_pruning_masks"
            ] = resized_weights_ckpt
            params["sparsification_params"]["resize_pruned_matrices"] = True
        else:
            with self.graph.as_default():
                saver = tf.train.Saver()
                trained_weights_ckpt = saver.save(
                    self.session, self.tmp_ckpt_name, write_meta_graph=False
                )
            self.warm_start_checkpoint = trained_weights_ckpt

        # inference graph
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.variable_scope("input_feeding", reuse=tf.AUTO_REUSE):
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

            self.input_tensors["probabilities"] = log_probs
            self.input_tensors["predictions"] = predictions

            trainable_vars = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            assignment_map = {
                v.name.split(":")[0]: v.name.split(":")[0] for v in trainable_vars
            }
            tf.train.init_from_checkpoint(
                self.warm_start_checkpoint, assignment_map=assignment_map
            )

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

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

        start = time.time()
        if self.tflite_quantise:
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
            log_probs = self.interpreter.get_tensor(self.out_index)
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
                    self.important_tensors["segment_ids"]: np.array(
                        example.segment_ids
                    ).reshape(-1, self.max_seq_length),
                },
            )
        print ("inference time: {:.3f}".format(time.time() - start))

        probabilities = list(np.exp(log_probs)[0])

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

        if meta["tflite_quantise"]:
            tflite_model_file = "tflite/converted_model_bert.tflite"
            tflite_input_tensors = [
                input_tensors[name]
                for name in ["input_ids", "input_mask", "segment_ids"]
            ]
            tflite_output_tensors = [input_tensors[name] for name in ["probabilities"]]
            converter = tf.lite.TFLiteConverter.from_session(
                sess, tflite_input_tensors, tflite_output_tensors
            )

            converter.optimizations = [
                # tf.lite.Optimize.DEFAULT
                # tf.lite.Optimize.OPTIMIZE_FOR_SIZE
                # tf.lite.Optimize.OPTIMIZE_FOR_LATENCY
            ]

            ## Representative dataset for full quantisation with TFLite
            ## representative dataset: generator, each element is a list of input items,
            ## items are passed to in_tensors in the order established when creating the converter
            ## https://github.com/tensorflow/tensorflow/blob/d883916ee45f3af81e81eefb7e8495d9fab6d231/tensorflow/lite/python/optimize/calibration_wrapper.cc#L110
            """
            with open('data/calibration_data.pickle', 'rb') as handle:
                calibration_data = pickle.load(handle)
            def representative_dataset_gen():
                for d in calibration_data:
                    yield d
            converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
            """

            ## https://github.com/tensorflow/tensorflow/blob/61128913681a016033143fbe9b60140d983b3c98/tensorflow/lite/tools/optimize/quantize_model.cc
            tflite_model = converter.convert()
            open(tflite_model_file, "wb").write(tflite_model)

            obj = cls(
                component_config=meta,
                session=sess,
                label_list=label_list,
                predict_fn=predict_fn,
                tflite_quantise=True,
            )

            obj.interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
            obj.interpreter.allocate_tensors()
            obj.in_indices = {}
            for i, in_name in enumerate(["input_ids", "input_mask", "segment_ids"]):
                obj.in_indices[in_name] = obj.interpreter.get_input_details()[i][
                    "index"
                ]
            obj.out_index = obj.interpreter.get_output_details()[0]["index"]
            return obj

        return cls(
            component_config=meta,
            session=sess,
            label_list=label_list,
            tflite_quantise=False,
            important_tensors=input_tensors,
        )
