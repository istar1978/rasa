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
    """Intent classifier using BERT.
    """

    name = "BertIntentClassifier"

    provides = ["intent", "intent_ranking"]

    defaults = {
        "batch_size": 64,
        "epochs": 2,
        "learning_rate": 2e-5,
        "max_seq_length": 128,
        "warmup_proportion": 0.1,
        "save_checkpoints_steps": 1000,
        "save_summary_steps": 500,
        "bert_tfhub_module_handle": "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        "pretrained_model_dir": None,
        "checkpoint_dir": "./tmp/bert",
        "checkpoint_remove_before_training": True,
    }

    def _load_bert_params(self, config: Dict[Text, Any]) -> None:
        self.bert_tfhub_module_handle = config["bert_tfhub_module_handle"]
        self.pretrained_model_dir = config["pretrained_model_dir"]

        if self.pretrained_model_dir:
            dir_files = os.listdir(self.pretrained_model_dir)
            print(dir_files)
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

    def _load_params(self) -> None:
        self._load_bert_params(self.component_config)
        self._load_train_params(self.component_config)

    def __init__(
        self,
        component_config=None,
        session: Optional["tf.Session"] = None,
        label_list: Optional[np.ndarray] = None,
        predict_fn: Optional["Predictor"] = None,
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
        if self.epochs <= 0:
            num_train_steps = 1
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        tf.logging.info("***** Running training *****")
        tf.logging.info("Num examples = %d", len(train_examples))
        tf.logging.info("Batch size = %d", self.batch_size)
        tf.logging.info("Num steps = %d", num_train_steps)
        tf.logging.info("Num epochs = %d", self.epochs)
        train_features = convert_examples_to_features(
            train_examples, self.label_list, self.max_seq_length, self.tokenizer
        )

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

        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn, 
            config=run_config, 
            params={"batch_size": self.batch_size},
            model_dir=self.checkpoint_dir
        )

        # Start training
        self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        self.session = tf.Session()

        # Create predictor incase running evaluation
        self.predict_fn = predictor.from_estimator(
            self.estimator, serving_input_fn_builder(self.max_seq_length)
        )

    def process(self, message: Message, **kwargs: Any) -> None:
        """Return the most likely intent and its similarity to the input"""

        # Classifier needs this to be non empty, so we set to first label.
        message.data["intent"] = self.label_list[0]
        # print("Message: {}".format(message.text))
        predict_examples = get_test_examples([message])
        # print("Examples:")
        # [print(e.guid, e.text_a, e.text_b, e.label) for e in predict_examples]
        predict_features = convert_examples_to_features(
            predict_examples, self.label_list, self.max_seq_length, self.tokenizer
        )
        # print("Features")
        # [print("input ids", e.input_ids, "input mask", e.input_mask, "segment ids", e.segment_ids, "label id", e.label_id, e.is_real_example) for e in predict_features]

        # Get first index since we are only classifying text blob at a time.
        example = predict_features[0]

        use_tflite = True
        
        # start = time.time()
        if use_tflite:
            
            input_ids = np.array(example.input_ids).reshape(-1, self.max_seq_length).astype(np.int32)
            # print(input_ids.shape)
            self.interpreter.set_tensor(
                self.in_indices["input_ids"], 
                input_ids)

            input_mask = np.array(example.input_mask).reshape(-1, self.max_seq_length).astype(np.int32)
            self.interpreter.set_tensor(
                self.in_indices["input_mask"], 
                input_mask)

            label_ids = np.array(example.label_id).reshape(-1).astype(np.int32)
            self.interpreter.set_tensor(
                self.in_indices["label_ids"], 
                label_ids)

            segment_ids = np.array(example.segment_ids).reshape(-1, self.max_seq_length).astype(np.int32)
            self.interpreter.set_tensor(
                self.in_indices["segment_ids"], 
                segment_ids)                
            
            self.interpreter.invoke()
            result = self.interpreter.get_tensor(self.out_index)
            probabilities = list(np.exp(result[0]))        
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
        # print("inference time: {:.3f}".format(time.time() - start))
        # full inference time: 116.083s (0.1176s/message)
        
        # print(result)

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
        # print(intent)
        # exit(0)

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
            model_dir, 
            serving_input_fn_builder(self.max_seq_length)
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
        tflite_model_file = "tflite/converted_model_bert.tflite"
        
        if model_dir and meta.get("model_path"):
            
            with io.open(
                os.path.join(model_dir, cls.name + "_label_list.pkl"), "rb"
            ) as f:
                label_list = pickle.load(f)

            model_path = os.path.join(model_dir, [p for p in os.walk(model_dir)][0][1][0])
            print(os.listdir(model_path))
            # converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
            # tflite_model = converter.convert()
            # open("converted_model.tflite", "wb").write(tflite_model)

            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session()
                predict_fn = predictor.from_saved_model(model_path)
                [print(t) for t in predict_fn.feed_tensors]
                [print(t) for t in predict_fn.fetch_tensors]
                print(predict_fn.graph)
                
                # writer = tf.summary.FileWriter(logdir="tfgraph-full", graph=predict_fn.graph)
                # writer.flush()
                # exit()
                in_tensor_names = ["input_ids", "input_mask", "label_ids", "segment_ids"]
                # """
                in_tensors = [predict_fn.feed_tensors[name] for name in in_tensor_names]
                out_tensors = [t for _, t in predict_fn.fetch_tensors.items()]
                converter = tf.lite.TFLiteConverter.from_session(
                    predict_fn.session, in_tensors, out_tensors
                )

                converter.optimizations = [
                    # tf.lite.Optimize.DEFAULT # 4s (~219it/s) (4.604s invoking time)
                    # tf.lite.Optimize.OPTIMIZE_FOR_SIZE # 4s (~220it/s) (4.707s invoking time)
                    tf.lite.Optimize.OPTIMIZE_FOR_LATENCY # 4s (~215it/s) (4.630s invoking time)
                ]
                tflite_model = converter.convert()
                open(tflite_model_file, "wb").write(tflite_model)
                # """

                obj = cls(
                    component_config=meta,
                    session=sess,
                    label_list=label_list,
                    predict_fn=predict_fn,
                )

                obj.interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
                obj.interpreter.allocate_tensors()
                # exit()
                obj.in_indices = {}
                for i, in_name in enumerate(in_tensor_names):
                    print(in_name, obj.interpreter.get_input_details()[i]["shape"])
                    obj.in_indices[in_name] = obj.interpreter.get_input_details()[i]["index"]
                # obj.interpreter.set_tensor(obj.interpreter.get_input_details()[1]["index"], 
                #                            obj.all_intents_embed_values)

                obj.out_index = obj.interpreter.get_output_details()[0]["index"]
                print(obj.in_indices, obj.out_index)
                # exit()
                return obj

        else:
            logger.warning(
                "Failed to load nlu model. Maybe path {} "
                "doesn't exist"
                "".format(os.path.abspath(model_dir))
            )
            return cls(component_config=meta)
