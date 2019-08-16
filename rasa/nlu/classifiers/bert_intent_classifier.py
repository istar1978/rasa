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
    # model_fn_builder,
    build_input_dataset,
    build_model,
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
        "load_pruning_masks_from_checkpoint": False,
        "resize_pruned_matrices": False,
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

    def _train_tf_dataset(
        self,
        train_init_op,
        # val_init_op,
        batch_size_in,
        batch_size,
        loss: "tf.Tensor",
        # is_training: "tf.Tensor",
        train_op: "tf.Tensor",
        num_train_steps=None,
    ) -> None:
        """Train tf graph"""

        self.session.run(tf.global_variables_initializer())
        print ("INITIALISED")
        # if self.evaluate_on_num_examples:
        #     logger.info(
        #         "Accuracy is updated every {} epochs"
        #         "".format(self.evaluate_every_num_epochs)
        #     )
        print ("TRAINING...")
        pbar = tqdm(
            range(max(self.epochs, 1)), desc="Epochs"
        )  # , disable=is_logging_disabled())
        train_acc = 0
        last_loss = 0
        for ep in pbar:

            # batch_size = self._linearly_increasing_batch_size(ep)

            self.session.run(train_init_op, feed_dict={batch_size_in: batch_size})

            ep_loss = 0
            batches_per_epoch = 0
            while num_train_steps is None or batches_per_epoch <= num_train_steps:
                try:
                    _, batch_loss = self.session.run((train_op, loss), feed_dict={})

                except tf.errors.OutOfRangeError:
                    break

                batches_per_epoch += 1
                ep_loss += batch_loss

            ep_loss /= batches_per_epoch

            # if self.evaluate_on_num_examples and val_init_op is not None:
            #     if (
            #         ep == 0
            #         or (ep + 1) % self.evaluate_every_num_epochs == 0
            #         or (ep + 1) == self.epochs
            #     ):
            #         train_acc = self._output_training_stat_dataset(val_init_op)
            #         last_loss = ep_loss

            #     pbar.set_postfix(
            #         {
            #             "loss": "{:.3f}".format(ep_loss),
            #             "acc": "{:.3f}".format(train_acc),
            #         }
            #     )
            # else:
            pbar.set_postfix({"loss": "{:.3f}".format(ep_loss)})

        # if self.evaluate_on_num_examples:
        #     logger.info(
        #         "Finished training embedding classifier, "
        #         "loss={:.3f}, train accuracy={:.3f}"
        #         "".format(last_loss, train_acc)
        #     )

    def train(self, training_data, cfg, **kwargs):
        """Train this component."""

        # Clean up checkpoint
        if self.checkpoint_remove_before_training and os.path.exists(
            self.checkpoint_dir
        ):
            shutil.rmtree(self.checkpoint_dir, ignore_errors=True)

        self.label_list = get_labels(training_data)

        # run_config = tf.estimator.RunConfig(
        #     model_dir=self.checkpoint_dir,
        #     keep_checkpoint_max=30,
        #     save_summary_steps=self.save_summary_steps,
        #     save_checkpoints_steps=self.save_checkpoints_steps,
        # )

        do_training = False

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

        ###############################################################################
        # Building the model without tf.Estimator
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
            # print("#####################", minibatch)
            # exit(0)
            train_init_op = iterator.make_initializer(train_dataset)
            train_op, loss = build_model(
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
                logdir="tfgraph-bert-np-train-new", graph=self.graph
            )
            writer.flush()

            # [print(v.name) for v in self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
            trainable_vars = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            if self.hat_layer_in_checkpoint:
                assignment_map = {
                    v.name.split(":")[0]: v.name.split(":")[0] for v in trainable_vars
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
                num_train_steps=(None if self.epochs > 0 else num_train_steps),
            )
            # exit(0)

            # create placeholders for the prediction graph
            self.input_placeholders = {}
            for input_name, input_tensor in minibatch.items():
                print (input_name, input_tensor)
                placeholder = tf.placeholder(
                    input_tensor.dtype, input_tensor.shape, name=input_name
                )
                self.input_placeholders[input_name] = placeholder

            """
            predictions = build_model(features=self.input_placeholders,
                                    mode="predict",
                                    params=params,
                                    bert_tfhub_module_handle=self.bert_tfhub_module_handle,
                                    num_labels=len(self.label_list),
                                    learning_rate=self.learning_rate,
                                    num_train_steps=num_train_steps,
                                    num_warmup_steps=num_warmup_steps,
                                    bert_config=bert_config)
            
            writer = tf.summary.FileWriter(logdir="tfgraph-bert-np-test-new", graph=self.graph)
            writer.flush()
            print("GRAPH WRITTEN")
            print(predictions)
            exit(0)
            """

            # exit(0)
        ###############################################################################

        """
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

        warm_start_settings = None
        if self.warm_start_checkpoint:
            if self.hat_layer_in_checkpoint:
                if not self.load_pruning_masks_from_checkpoint:
                    warm_start_settings = tf.estimator.WarmStartSettings(
                        self.warm_start_checkpoint
                    )
                else:
                    warm_start_settings = tf.estimator.WarmStartSettings(
                        self.warm_start_checkpoint, vars_to_warm_start=[".*"]
                    )
            else:
                warm_start_settings = tf.estimator.WarmStartSettings(
                    self.warm_start_checkpoint, vars_to_warm_start="bert.*"
                )
        print("ESTIMATOR BUILT")
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params=params,
            model_dir=self.checkpoint_dir,
            warm_start_from=warm_start_settings,
        )
        # print(self.estimator)
        # exit(0)

        # Start training
        if do_training:
            self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        # exit(0)

        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        print("SESSION CREATED")

        serving_input_receiver_fn = serving_input_fn_builder(self.max_seq_length, is_predicting=True)
        print("SERVING INPUT RECEIVER CREATED")

        # Create predictor incase running evaluation
        self.predict_fn = predictor.from_estimator(
            self.estimator, serving_input_receiver_fn
        )
        print("PREDICTOR CREATED")


        g = self.predict_fn.graph
        writer = tf.summary.FileWriter(logdir="tfgraph-bert-test-np", graph=g)
        writer.flush()
        print("GRAPH WRITTEN")

        # exit(0)
        """

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

        with self.graph.as_default():
            # 'input_ids': <tf.Tensor 'IteratorGetNext:0' shape=(1, 128) dtype=int32>,
            # 'input_mask': <tf.Tensor 'IteratorGetNext:1' shape=(1, 128) dtype=int32>,
            # 'segment_ids': <tf.Tensor 'IteratorGetNext:3' shape=(1, 128) dtype=int32>,
            # 'label_ids': <tf.Tensor 'IteratorGetNext:2' shape=(1,) dtype=int32>}

            # for placeholder_name, placeholder_tensor in self.input_placeholders.items():
            # self.graph.clear_collection()
            # self.graph.add_to_collection("message_placeholder", self.a_in)

            # self.graph.clear_collection("intent_placeholder")
            # self.graph.add_to_collection("intent_placeholder", self.b_in)

            # self.graph.clear_collection("similarity_op")
            # self.graph.add_to_collection("similarity_op", self.sim_op)

            # self.graph.clear_collection("all_intents_embed_in")
            # self.graph.add_to_collection(
            #     "all_intents_embed_in", self.all_intents_embed_in
            # )
            # self.graph.clear_collection("sim_all")
            # self.graph.add_to_collection("sim_all", self.sim_all)

            # self.graph.clear_collection("word_embed")
            # self.graph.add_to_collection("word_embed", self.word_embed)
            # self.graph.clear_collection("intent_embed")
            # self.graph.add_to_collection("intent_embed", self.intent_embed)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        # placeholder_dims = {
        #     "a_in": np.int(self.a_in.shape[-1]),
        #     "b_in": np.int(self.b_in.shape[-1]),
        # }
        # with io.open(
        #     os.path.join(model_dir, file_name + "_placeholder_dims.pkl"), "wb"
        # ) as f:
        #     pickle.dump(placeholder_dims, f)
        # with io.open(
        #     os.path.join(model_dir, file_name + "_inv_intent_dict.pkl"), "wb"
        # ) as f:
        #     pickle.dump(self.inv_intent_dict, f)
        # with io.open(
        #     os.path.join(model_dir, file_name + "_encoded_all_intents.pkl"), "wb"
        # ) as f:
        #     pickle.dump(self.encoded_all_intents, f)
        # with io.open(
        #     os.path.join(model_dir, file_name + "_all_intents_embed_values.pkl"), "wb"
        # ) as f:
        #     pickle.dump(self.all_intents_embed_values, f)

        return {"file": file_name}

        ##########################################

        """
        # try:
        #     os.makedirs(model_dir)
        # except OSError as e:
        #     # Be happy if someone already created the path
        #     import errno

        #     if e.errno != errno.EEXIST:
        #         raise
        print("SAVING ESTIMATOR")
        model_path = self.estimator.export_saved_model(
            model_dir, serving_input_fn_builder(self.max_seq_length, is_predicting=True)
        )
        print("ESTIMATOR SAVED")

        with io.open(os.path.join(model_dir, self.name + "_label_list.pkl"), "wb") as f:
            pickle.dump(self.label_list, f)

        return {"model_path": model_path.decode("UTF-8")}
        """

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: "Metadata" = None,
        cached_component: Optional["EmbeddingIntentClassifier"] = None,
        **kwargs: Any
    ) -> "BertIntentClassifier":
        if not (model_dir and meta.get("file")):
            logger.warning(
                "Failed to load nlu model. Maybe path {} "
                "doesn't exist"
                "".format(os.path.abspath(model_dir))
            )
            return cls(component_config=meta)

        file_name = meta.get("file")
        checkpoint = os.path.join(model_dir, file_name + ".ckpt")

        with io.open(
            os.path.join(model_dir, file_name + "_inv_intent_dict.pkl"), "rb"
        ) as f:
            inv_intent_dict = pickle.load(f)
        with io.open(
            os.path.join(model_dir, file_name + "_encoded_all_intents.pkl"), "rb"
        ) as f:
            encoded_all_intents = pickle.load(f)
        with io.open(
            os.path.join(model_dir, file_name + "_all_intents_embed_values.pkl"), "rb"
        ) as f:
            all_intents_embed_values = pickle.load(f)

        graph = tf.Graph()
        with graph.as_default():
            print ("loading...")
            sess = tf.Session()
            if meta["gpu_lstm"]:
                # rebuild tf graph for prediction
                with io.open(
                    os.path.join(model_dir, file_name + "_placeholder_dims.pkl"), "rb"
                ) as f:
                    placeholder_dims = pickle.load(f)
                reg = tf.contrib.layers.l2_regularizer(meta["C2"])

                a_in = tf.placeholder(
                    tf.float32, (None, None, placeholder_dims["a_in"]), name="a"
                )
                b_in = tf.placeholder(
                    tf.float32, (None, None, placeholder_dims["b_in"]), name="b"
                )
                a = cls._create_tf_gpu_predict_embed(
                    meta,
                    a_in,
                    meta["hidden_layers_sizes_a"],
                    name="a_and_b" if meta["share_embedding"] else "a",
                )
                word_embed = tf.layers.dense(
                    inputs=a,
                    units=meta["embed_dim"],
                    kernel_regularizer=reg,
                    name="embed_layer_{}".format("a"),
                    reuse=tf.AUTO_REUSE,
                )

                b = cls._create_tf_gpu_predict_embed(
                    meta,
                    b_in,
                    meta["hidden_layers_sizes_b"],
                    name="a_and_b" if meta["share_embedding"] else "b",
                )
                intent_embed = tf.layers.dense(
                    inputs=b,
                    units=meta["embed_dim"],
                    kernel_regularizer=reg,
                    name="embed_layer_{}".format("b"),
                    reuse=tf.AUTO_REUSE,
                )

                tiled_intent_embed = cls._tf_sample_neg(
                    intent_embed, None, None, tf.shape(word_embed)[0]
                )

                sim_op = cls._tf_gpu_sim(meta, word_embed, tiled_intent_embed)

                all_intents_embed_in = tf.placeholder(
                    tf.float32,
                    (None, None, meta["embed_dim"]),
                    name="all_intents_embed",
                )
                sim_all = cls._tf_gpu_sim(meta, word_embed, all_intents_embed_in)

                saver = tf.train.Saver()

            else:
                print ("not a gou lstm")
                saver = tf.train.import_meta_graph(checkpoint + ".meta")
                # Speed on Sara test data (using extremely deep BOW model):
                # 5s (~170it/s) using full model (6.605s invoking time)
                # 5s (~185it/s) using converted model without any optimisation (6.308s invoking time)
                convert = True
                if convert:
                    print ("converting")
                    a_in = tf.get_collection("message_placeholder")[0]
                    b_in = tf.get_collection("intent_placeholder")[0]

                    sim_op = tf.get_collection("similarity_op")[0]

                    all_intents_embed_in = tf.get_collection("all_intents_embed_in")[0]
                    sim_all = tf.get_collection("sim_all")[0]

                    word_embed = tf.get_collection("word_embed")[0]
                    intent_embed = tf.get_collection("intent_embed")[0]

                    num_intents = len([i for i in inv_intent_dict.items()])
                    all_intents_embed_in.set_shape(
                        (1, num_intents, all_intents_embed_in.shape[-1])
                    )
                    a_in.set_shape((1, a_in.shape[-1]))

                    saver.restore(sess, checkpoint)

                    in_tensors = [a_in, all_intents_embed_in]
                    out_tensors = [sim_all]
                    converter = tf.lite.TFLiteConverter.from_session(
                        sess, in_tensors, out_tensors
                    )

                    converter.optimizations = [
                        # tf.lite.Optimize.DEFAULT # 4s (~219it/s) (4.604s invoking time)
                        # tf.lite.Optimize.OPTIMIZE_FOR_SIZE # 4s (~220it/s) (4.707s invoking time)
                        # tf.lite.Optimize.OPTIMIZE_FOR_LATENCY # 4s (~215it/s) (4.630s invoking time)
                    ]
                    tflite_model = converter.convert()
                    open(tflite_model_file, "wb").write(tflite_model)

                    obj = cls(
                        component_config=meta,
                        inv_intent_dict=inv_intent_dict,
                        encoded_all_intents=encoded_all_intents,
                        all_intents_embed_values=all_intents_embed_values,
                        session=sess,
                        graph=graph,
                        message_placeholder=a_in,
                        intent_placeholder=None,
                        all_intents_embed_in=all_intents_embed_in,
                        sim_all=sim_all,
                        word_embed=None,
                        intent_embed=None,
                        is_tflite=True,
                        tflite_path=tflite_model_file,
                    )

                    obj.interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
                    obj.interpreter.allocate_tensors()

                    obj.a_in_index = obj.interpreter.get_input_details()[0]["index"]
                    obj.interpreter.set_tensor(
                        obj.interpreter.get_input_details()[1]["index"],
                        obj.all_intents_embed_values,
                    )

                    obj.sim_all_index = obj.interpreter.get_output_details()[0]["index"]

                    return obj
                else:
                    print ("doing the vanilla loading")
                    a_in = tf.get_collection("message_placeholder")[0]
                    b_in = tf.get_collection("intent_placeholder")[0]

                    sim_op = tf.get_collection("similarity_op")[0]

                    all_intents_embed_in = tf.get_collection("all_intents_embed_in")[0]
                    sim_all = tf.get_collection("sim_all")[0]

                    word_embed = tf.get_collection("word_embed")[0]
                    intent_embed = tf.get_collection("intent_embed")[0]
                    saver.restore(sess, checkpoint)

        #############################################################################

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
