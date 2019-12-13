import collections
import logging
import pickle
import warnings
import os
import typing
import numpy as np
from typing import Any, Dict, List, Optional, Text, Tuple, Union, NamedTuple

from tf_metrics import f1

from rasa.nlu.config import InvalidConfigError, RasaNLUModelConfig
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    MESSAGE_TOKENS_NAMES,
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_VECTOR_DENSE_FEATURE_NAMES,
    MESSAGE_SPACY_FEATURES_NAMES,
    MESSAGE_ENTITIES_ATTRIBUTE,
)
from rasa.constants import DOCS_BASE_URL

import tensorflow as tf
import rasa.utils.train_utils as train_utils
import rasa.utils.io as io_utils

try:
    import spacy
except ImportError:
    spacy = None

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from sklearn_crfsuite import CRF
    from spacy.tokens import Doc


class CRFToken(NamedTuple):
    text: Text
    tag: Text
    entity: Text
    pattern: Dict[Text, Any]
    dense_features: np.ndarray


class CRFEntityExtractor(EntityExtractor):

    provides = [MESSAGE_ENTITIES_ATTRIBUTE]

    requires = [MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE]]

    defaults = {
        # BILOU_flag determines whether to use BILOU tagging or not.
        # More rigorous however requires more examples per entity
        # rule of thumb: use only if more than 100 egs. per entity
        "BILOU_flag": True,
        # crf_features is [before, word, after] array with before, word,
        # after holding keys about which
        # features to use for each word, for example, 'title' in
        # array before will have the feature
        # "is the preceding word in title case?"
        # POS features require spaCy to be installed
        "features": [
            ["low", "title", "upper"],
            [
                "bias",
                "low",
                "prefix5",
                "prefix2",
                "suffix5",
                "suffix3",
                "suffix2",
                "upper",
                "title",
                "digit",
                "pattern",
            ],
            ["low", "title", "upper"],
        ],
        # training parameters
        # initial and final batch sizes - batch size will be
        # linearly increased for each epoch
        "batch_size": [64, 256],
        # how to create batches
        "batch_strategy": "balanced",  # string 'sequence' or 'balanced'
        # number of epochs
        "epochs": 300,
        # set random seed to any int to get reproducible results
        "random_seed": None,
        # whether the loss should be normalized or not
        "normalize_loss": False,
        # embedding parameters
        # default dense dimension used if no dense features are present
        "dense_dim": 512,
        # regularization parameters
        # the scale of L2 regularization
        "C2": 0.002,
        # the scale of how critical the algorithm should be of minimizing the
        # maximum similarity between embeddings of different labels
        "C_emb": 0.8,
        # dropout rate for rnn
        "droprate": 0.2,
        # visualization of accuracy
        # how often to calculate training accuracy
        "evaluate_every_num_epochs": 20,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        "evaluate_on_num_examples": 0,  # large values may hurt performance
    }

    function_dict = {
        "low": lambda crf_token: crf_token.text.lower(),  # pytype: disable=attribute-error
        "title": lambda crf_token: crf_token.text.istitle(),  # pytype: disable=attribute-error
        "prefix5": lambda crf_token: crf_token.text[:5],
        "prefix2": lambda crf_token: crf_token.text[:2],
        "suffix5": lambda crf_token: crf_token.text[-5:],
        "suffix3": lambda crf_token: crf_token.text[-3:],
        "suffix2": lambda crf_token: crf_token.text[-2:],
        "suffix1": lambda crf_token: crf_token.text[-1:],
        "pos": lambda crf_token: crf_token.tag,
        "pos2": lambda crf_token: crf_token.tag[:2],
        "bias": lambda crf_token: "bias",
        "upper": lambda crf_token: crf_token.text.isupper(),  # pytype: disable=attribute-error
        "digit": lambda crf_token: crf_token.text.isdigit(),  # pytype: disable=attribute-error
        "pattern": lambda crf_token: crf_token.pattern,
        "text_dense_features": lambda crf_token: crf_token.dense_features,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        ent_tagger: Optional["CRF"] = None,
        feature_id_dict: Optional[Dict[Text, Dict[int, Text]]] = None,
        inverted_tag_dict: Optional[Dict[int, Text]] = None,
        session: Optional["tf.Session"] = None,
        graph: Optional["tf.Graph"] = None,
        batch_placeholder: Optional["tf.Tensor"] = None,
        entity_prediction: Optional["tf.Tensor"] = None,
        batch_tuple_sizes: Optional[Dict] = None,
        attention_weights: Optional["tf.Tensor"] = None,
    ) -> None:

        super().__init__(component_config)

        self.ent_tagger = ent_tagger

        self._validate_configuration()

        self._check_pos_features_and_spacy()

        self._tf_config = train_utils.load_tf_config(self.component_config)

        self.load_tf_params(self.component_config)

        # transform numbers to labels
        self.feature_id_dict = feature_id_dict
        self.inverted_tag_dict = inverted_tag_dict

        # tf related instances
        self.session = session
        self.graph = graph
        self.batch_in = batch_placeholder
        self.entity_prediction = entity_prediction

        # keep the input tuple sizes in self.batch_in
        self.batch_tuple_sizes = batch_tuple_sizes

        # internal tf instances
        self._iterator = None
        self._train_op = None
        self._is_training = None
        self._in_layer_norm = {}

        # number of entity tags
        self.num_tags = 0

        self.attention_weights = attention_weights

    def load_tf_params(self, config: Optional[Dict[Text, Any]]):

        self.batch_in_size = config["batch_size"]
        self.batch_in_strategy = config["batch_strategy"]

        self.normalize_loss = config["normalize_loss"]
        self.epochs = config["epochs"]

        self.dense_dim = config["dense_dim"]

        self.random_seed = self.component_config["random_seed"]

        self.C2 = config["C2"]
        self.C_emb = config["C_emb"]
        self.droprate = config["droprate"]
        self.evaluate_every_num_epochs = config["evaluate_every_num_epochs"]
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs
        self.evaluate_on_num_examples = config["evaluate_on_num_examples"]

        self.training_log_file = io_utils.create_temporary_file("")

    def _check_pos_features_and_spacy(self):
        import itertools

        features = self.component_config.get("features", [])
        fts = set(itertools.chain.from_iterable(features))
        self.pos_features = "pos" in fts or "pos2" in fts
        if self.pos_features:
            self._check_spacy()

    def _validate_configuration(self):
        if len(self.component_config.get("features", [])) % 2 != 1:
            raise ValueError(
                "Need an odd number of crf feature lists to have a center word."
            )

    @staticmethod
    def _check_spacy():
        if spacy is None:
            raise ImportError(
                "Failed to import `spaCy`. "
                "`spaCy` is required for POS features "
                "See https://spacy.io/usage/ for installation"
                "instructions."
            )

    @classmethod
    def required_packages(cls):
        return ["sklearn_crfsuite", "sklearn"]

    def train(
        self, training_data: "TrainingData", config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:
        """Train the embedding label classifier on a data set."""

        if not training_data.entity_examples:
            return

        self._check_spacy_doc(training_data.training_examples[0])

        # set numpy random seed
        np.random.seed(self.random_seed)

        dataset = self._create_dataset(training_data.training_examples)
        session_data = self.preprocess_train_data(dataset)

        if self.evaluate_on_num_examples:
            session_data, eval_session_data = train_utils.train_val_split(
                session_data,
                self.evaluate_on_num_examples,
                self.random_seed,
                label_key="label_ids",
            )
        else:
            eval_session_data = None

        self.graph = tf.Graph()
        with self.graph.as_default():
            # set random seed
            tf.set_random_seed(self.random_seed)

            # allows increasing batch size
            batch_size_in = tf.placeholder(tf.int64)

            (
                self._iterator,
                train_init_op,
                eval_init_op,
            ) = train_utils.create_iterator_init_datasets(
                session_data,
                eval_session_data,
                batch_size_in,
                "balanced",
                label_key="label_ids",
            )

            self._is_training = tf.placeholder_with_default(False, shape=())

            metrics = self._build_tf_train_graph(session_data)

            # calculate overall loss
            if self.normalize_loss:
                loss = tf.add_n(
                    [
                        _loss / (tf.stop_gradient(_loss) + 1e-8)
                        for _loss in metrics.loss.values()
                    ]
                )
            else:
                loss = tf.add_n(list(metrics.loss.values()))

            # define which optimizer to use
            self._train_op = tf.train.AdamOptimizer().minimize(loss)

            # train tensorflow graph
            self.session = tf.Session(config=self._tf_config)

            train_utils.train_tf_dataset(
                train_init_op,
                eval_init_op,
                batch_size_in,
                metrics,
                self._train_op,
                self.session,
                self._is_training,
                self.epochs,
                self.batch_in_size,
                self.evaluate_on_num_examples,
                self.evaluate_every_num_epochs,
                output_file=self.training_log_file,
            )

            # rebuild the graph for prediction
            self._build_tf_pred_graph(session_data)

    def process(self, message: Message, **kwargs: Any) -> None:

        self._check_spacy_doc(message)

        extracted = self.add_extractor_name(self.extract_entities(message))
        message.set(
            MESSAGE_ENTITIES_ATTRIBUTE,
            message.get(MESSAGE_ENTITIES_ATTRIBUTE, []) + extracted,
            add_to_output=True,
        )

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component: Optional["CRFEntityExtractor"] = None,
        **kwargs: Any,
    ) -> "CRFEntityExtractor":
        if model_dir and meta.get("file"):
            file_name = meta.get("file")
            checkpoint = os.path.join(model_dir, file_name + ".ckpt")

            with open(os.path.join(model_dir, file_name + ".tf_config.pkl"), "rb") as f:
                _tf_config = pickle.load(f)

            graph = tf.Graph()
            with graph.as_default():
                session = tf.compat.v1.Session(config=_tf_config)
                saver = tf.compat.v1.train.import_meta_graph(checkpoint + ".meta")

                saver.restore(session, checkpoint)

                batch_in = train_utils.load_tensor("batch_placeholder")
                entity_prediction = train_utils.load_tensor("entity_prediction")

                attention_weights = train_utils.load_tensor("attention_weights")

            with open(
                os.path.join(model_dir, file_name + ".feature_id_dict.pkl"), "rb"
            ) as f:
                feature_id_dict = pickle.load(f)

            with open(
                os.path.join(model_dir, file_name + ".inv_tag_dict.pkl"), "rb"
            ) as f:
                inv_tag_dict = pickle.load(f)

            with open(
                os.path.join(model_dir, file_name + ".batch_tuple_sizes.pkl"), "rb"
            ) as f:
                batch_tuple_sizes = pickle.load(f)

            return cls(
                component_config=meta,
                feature_id_dict=feature_id_dict,
                inverted_tag_dict=inv_tag_dict,
                session=session,
                graph=graph,
                batch_placeholder=batch_in,
                entity_prediction=entity_prediction,
                attention_weights=attention_weights,
                batch_tuple_sizes=batch_tuple_sizes,
            )

        else:
            warnings.warn(
                f"Failed to load nlu model. Maybe path '{os.path.abspath(model_dir)}' "
                "doesn't exist."
            )
            return cls(component_config=meta)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
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
            train_utils.persist_tensor("batch_placeholder", self.batch_in, self.graph)
            train_utils.persist_tensor(
                "entity_prediction", self.entity_prediction, self.graph
            )
            train_utils.persist_tensor(
                "attention_weights", self.attention_weights, self.graph
            )

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with open(
            os.path.join(model_dir, file_name + ".feature_id_dict.pkl"), "wb"
        ) as f:
            pickle.dump(self.feature_id_dict, f)

        with open(os.path.join(model_dir, file_name + ".inv_tag_dict.pkl"), "wb") as f:
            pickle.dump(self.inverted_tag_dict, f)

        with open(os.path.join(model_dir, file_name + ".tf_config.pkl"), "wb") as f:
            pickle.dump(self._tf_config, f)

        with open(
            os.path.join(model_dir, file_name + ".batch_tuple_sizes.pkl"), "wb"
        ) as f:
            pickle.dump(self.batch_tuple_sizes, f)

        return {"file": file_name}

    # TENSORFLOW METHODS

    def _build_tf_train_graph(
        self, session_data: train_utils.SessionDataType
    ) -> train_utils.TrainingMetrics:

        # get in tensors from generator
        self.batch_in = self._iterator.get_next()

        # convert batch format into sparse and dense tensors
        batch_data, _ = train_utils.batch_to_session_data(self.batch_in, session_data)

        mask = batch_data["text_mask"][0]
        a = self.combine_sparse_dense_features(
            batch_data["text_features"], mask, "text", sparse_dropout=True
        )
        c = self.combine_sparse_dense_features(batch_data["tag_ids"], mask, "tag")

        a = train_utils.create_tf_fnn(
            a,
            [128],
            self.droprate,
            self.C2,
            self._is_training,
            layer_name_suffix="text",
        )

        mask_up_to_last = 1 - tf.cumprod(1 - mask, axis=1, exclusive=True, reverse=True)

        sequence_lengths = tf.cast(tf.reduce_sum(mask_up_to_last[:, :, 0], 1), tf.int32)
        sequence_lengths.set_shape([mask.shape[0]])

        c = tf.reduce_sum(tf.nn.relu(c), -1)
        c = tf.cast(c, tf.int32)

        # CRF
        crf_params, logits, pred_ids = self._create_crf(a, sequence_lengths)

        # Loss
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, c, sequence_lengths, crf_params
        )
        loss = tf.reduce_mean(-log_likelihood)

        # calculate f1 score for train predictions
        pos_tag_indices = [k for k, v in self.inverted_tag_dict.items() if v != "O"]
        f1_score = f1(c, pred_ids, self.num_tags, pos_tag_indices, mask_up_to_last)

        metrics = train_utils.TrainingMetrics(loss={}, score={})
        metrics.loss["e_loss"] = loss
        metrics.score["e_f1"] = f1_score[1]
        return metrics

    def _create_crf(
        self, input: tf.Tensor, sequence_lengths: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.variable_scope("ner", reuse=tf.AUTO_REUSE):
            logits = train_utils.create_tf_embed(
                input, self.num_tags, self.C2, "crf-logits"
            )
            crf_params = tf.get_variable(
                "crf-params",
                [self.num_tags, self.num_tags],
                dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(self.C2),
            )
            pred_ids, _ = tf.contrib.crf.crf_decode(
                logits, crf_params, sequence_lengths
            )

            return crf_params, logits, pred_ids

    def _build_tf_pred_graph(self, session_data: train_utils.SessionDataType):

        shapes, types = train_utils.get_shapes_types(session_data)

        batch_placeholder = []
        for s, t in zip(shapes, types):
            batch_placeholder.append(tf.placeholder(t, s))

        self.batch_in = tf.tuple(batch_placeholder)

        batch_data, self.batch_tuple_sizes = train_utils.batch_to_session_data(
            self.batch_in, session_data
        )

        mask = batch_data["text_mask"][0]
        a = self.combine_sparse_dense_features(
            batch_data["text_features"], mask, "text"
        )

        a = train_utils.create_tf_fnn(
            a,
            [128],
            self.droprate,
            self.C2,
            self._is_training,
            layer_name_suffix="text",
        )

        mask_up_to_last = 1 - tf.cumprod(1 - mask, axis=1, exclusive=True, reverse=True)
        sequence_lengths = tf.cast(tf.reduce_sum(mask_up_to_last[:, :, 0], 1), tf.int32)

        # predict tagsx
        _, _, pred_ids = self._create_crf(a, sequence_lengths)
        self.entity_prediction = tf.to_int64(pred_ids)

    def combine_sparse_dense_features(
        self,
        features: List[Union["tf.Tensor", "tf.SparseTensor"]],
        mask: "tf.Tensor",
        name: Text,
        sparse_dropout: bool = False,
    ) -> "tf.Tensor":

        dense_features = []

        dense_dim = self.dense_dim
        # if dense features are present use the feature dimension of the dense features
        for f in features:
            if not isinstance(f, tf.SparseTensor):
                dense_dim = f.shape[-1]
                break

        for f in features:
            if isinstance(f, tf.SparseTensor):
                if sparse_dropout:
                    to_retain_prob = tf.random.uniform(
                        tf.shape(f.values), 0, 1, f.values.dtype
                    )
                    to_retain = tf.greater_equal(to_retain_prob, self.droprate)
                    _f = tf.sparse.retain(f, to_retain)
                    _f = tf.cond(self._is_training, lambda: _f, lambda: f)
                else:
                    _f = f

                dense_features.append(
                    train_utils.tf_dense_layer_for_sparse(
                        _f, dense_dim, name, self.C2, input_dim=int(f.shape[-1])
                    )
                )
            else:
                dense_features.append(f)

        return tf.concat(dense_features, axis=-1) * mask

    # CREATING DATASET / SESSION DATA

    def _create_dataset(self, examples: List[Message]) -> List[List[CRFToken]]:
        dataset = []

        for example in examples:
            entity_offsets = self._convert_example(example)
            dataset.append(self._from_json_to_crf(example, entity_offsets))

        return dataset

    def preprocess_train_data(self, dataset: List[List[CRFToken]]):
        """Prepares data for training."""
        y_train = [self._sentence_to_labels(sent) for sent in dataset]
        x_train = [self._sentence_to_features(sent) for sent in dataset]

        tag_id_dict = self._create_tag_id_dict(y_train)
        self.inverted_tag_dict = {v: k for k, v in tag_id_dict.items()}

        self.feature_id_dict = self._create_feature_id_dict(x_train)

        session_data = self._create_session_data(x_train, y_train, tag_id_dict)

        self.num_tags = len(self.inverted_tag_dict)

        return session_data

    @staticmethod
    def _create_tag_id_dict(tags: List[List[Text]]) -> Dict[Text, int]:
        """Create tag_id dictionary"""

        distinct_tags = {t for sent_tags in tags for t in sent_tags} - {"O"}

        tag_id_dict = {tag: idx for idx, tag in enumerate(sorted(distinct_tags), 1)}
        tag_id_dict["O"] = 0
        return tag_id_dict

    @staticmethod
    def _create_feature_id_dict(
        features: List[List[Dict[Text, Any]]]
    ) -> Dict[Text, Dict[Text, int]]:
        # build vocab of features
        vocab_x = collections.defaultdict(set)
        for sent_features in features:
            for token_features in sent_features:
                for key, val in token_features.items():
                    vocab_x[key].add(val)

        feature_id_dict = {}
        offset = 0
        for key, val in vocab_x.items():
            feature_id_dict[key] = {
                str(feature_val): idx
                for idx, feature_val in enumerate(sorted(val), offset)
            }
            offset += len(val)

        return feature_id_dict

    def _create_session_data(
        self,
        x_data: List[List[Dict[Text, Any]]],
        y_data: Optional[List[List[Text]]] = None,
        tag_id_dict: Optional[Dict[Text, int]] = None,
    ) -> train_utils.SessionDataType:
        """Prepare data for training and create a SessionDataType object"""
        tag_ids = []
        label_ids = []

        if y_data and tag_id_dict:
            for y in y_data:
                _tags = [tag_id_dict[_tag] for _tag in y]
                # transpose to have seq_len x 1
                tag_ids.append(np.array([_tags]).T)

            for y in y_data:
                entity_present = any([y != "O" for _tag in y])
                if entity_present:
                    label_ids.append(1)
                else:
                    label_ids.append(0)

        num_features = sum(
            [
                len(feature_vals.values())
                for feature_vals in self.feature_id_dict.values()
            ]
        )

        X_sparse = []
        # convert features into one-hot
        for x in x_data:
            x_features = []
            for crf_token in x:
                t_features = np.zeros(num_features)
                for k, v in crf_token.items():
                    if k in self.feature_id_dict and str(v) in self.feature_id_dict[k]:
                        idx = self.feature_id_dict[k][str(v)]
                        t_features[idx] = 1
                x_features.append(t_features)
            X_sparse.append(np.array(x_features))

        X_sparse = np.array(X_sparse)
        tag_ids = np.array(tag_ids)
        label_ids = np.array(label_ids)

        session_data = {}
        self._add_to_session_data(session_data, "text_features", [X_sparse])
        self._add_to_session_data(session_data, "tag_ids", [tag_ids])
        self._add_to_session_data(session_data, "label_ids", [label_ids])
        self._add_mask_to_session_data(session_data, "text_mask", "text_features")

        return session_data

    @staticmethod
    def _add_to_session_data(
        session_data: train_utils.SessionDataType, key: Text, features: List[np.ndarray]
    ):
        if not features:
            return

        session_data[key] = []

        for data in features:
            if data.size > 0:
                session_data[key].append(data)

    @staticmethod
    def _add_mask_to_session_data(
        session_data: train_utils.SessionDataType, key: Text, from_key: Text
    ):

        session_data[key] = []

        for data in session_data[from_key]:
            if data.size > 0:
                # explicitly add last dimension to mask
                # to track correctly dynamic sequences
                mask = np.array([np.ones((x.shape[0], 1)) for x in data])
                session_data[key].append(mask)
                break

    # HELPER METHODS

    def _check_spacy_doc(self, message):
        if (
            self.pos_features
            and message.get(MESSAGE_SPACY_FEATURES_NAMES[MESSAGE_TEXT_ATTRIBUTE])
            is None
        ):
            raise InvalidConfigError(
                "Could not find `spacy_doc` attribute for "
                "message {}\n"
                "POS features require a pipeline component "
                "that provides `spacy_doc` attributes, i.e. `SpacyNLP`. "
                "See {}/nlu/choosing-a-pipeline/#pretrained-embeddings-spacy "
                "for details".format(message.text, DOCS_BASE_URL)
            )

    @staticmethod
    def _convert_example(example: Message) -> List[Tuple[int, int, Text]]:
        def convert_entity(entity):
            return entity["start"], entity["end"], entity["entity"]

        return [
            convert_entity(ent) for ent in example.get(MESSAGE_ENTITIES_ATTRIBUTE, [])
        ]

    def extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Take a sentence and return entities in json format"""

        if self.session is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
            return []

        # create session data from message and convert it into a batch of 1
        text_data = self._from_text_to_crf(message)
        features = self._sentence_to_features(text_data)
        self.num_tags = len(self.inverted_tag_dict)
        session_data = self._create_session_data([features])
        batch = train_utils.prepare_batch(
            session_data, tuple_sizes=self.batch_tuple_sizes
        )

        # load tf graph and session
        predictions = self.session.run(
            self.entity_prediction,
            feed_dict={
                _x_in: _x for _x_in, _x in zip(self.batch_in, batch) if _x is not None
            },
        )

        tags = [self.inverted_tag_dict[p] for p in predictions[0]]

        if self.component_config["BILOU_flag"]:
            tags = [t[2:] if t[:2] in ["B-", "I-", "U-", "L-"] else t for t in tags]

        entities = self._convert_tags_to_entities(
            message.text, message.get("tokens", []), tags
        )

        extracted = self.add_extractor_name(entities)
        entities = message.get("entities", []) + extracted

        return entities

    def _convert_tags_to_entities(
        self, text: str, tokens: List[Token], tags: List[Text]
    ) -> List[Dict[Text, Any]]:
        entities = []
        last_tag = "O"
        for token, tag in zip(tokens, tags):
            if tag == "O":
                last_tag = tag
                continue

            # new tag found
            if last_tag != tag:
                entity = {
                    "entity": tag,
                    "start": token.offset,
                    "end": token.end,
                    "extractor": "flair",
                }
                entities.append(entity)

            # belongs to last entity
            elif last_tag == tag:
                entities[-1]["end"] = token.end

            last_tag = tag

        for entity in entities:
            entity["value"] = text[entity["start"] : entity["end"]]

        return entities

    def most_likely_entity(self, idx, entities):
        if len(entities) > idx:
            entity_probs = entities[idx]
        else:
            entity_probs = None
        if entity_probs:
            label = max(entity_probs, key=lambda key: entity_probs[key])
            if self.component_config["BILOU_flag"]:
                # if we are using bilou flags, we will combine the prob
                # of the B, I, L and U tags for an entity (so if we have a
                # score of 60% for `B-address` and 40% and 30%
                # for `I-address`, we will return 70%)
                return (
                    label,
                    sum([v for k, v in entity_probs.items() if k[2:] == label[2:]]),
                )
            else:
                return label, entity_probs[label]
        else:
            return "", 0.0

    def _create_entity_dict(
        self,
        message: Message,
        tokens: Union["Doc", List[Token]],
        start: int,
        end: int,
        entity: str,
        confidence: float,
    ) -> Dict[Text, Any]:
        if isinstance(tokens, list):  # tokens is a list of Token
            _start = tokens[start].offset
            _end = tokens[end].end
            value = tokens[start].text
            value += "".join(
                [
                    message.text[tokens[i - 1].end : tokens[i].offset] + tokens[i].text
                    for i in range(start + 1, end + 1)
                ]
            )
        else:  # tokens is a Doc
            _start = tokens[start].idx
            _end = tokens[start : end + 1].end_char
            value = tokens[start : end + 1].text

        return {
            "start": _start,
            "end": _end,
            "value": value,
            "entity": entity,
            "confidence": confidence,
        }

    @staticmethod
    def _entity_from_label(label):
        return label[2:]

    @staticmethod
    def _bilou_from_label(label):
        if len(label) >= 2 and label[1] == "-":
            return label[0].upper()
        return None

    def _find_bilou_end(self, word_idx, entities):
        ent_word_idx = word_idx + 1
        finished = False

        # get information about the first word, tagged with `B-...`
        label, confidence = self.most_likely_entity(word_idx, entities)
        entity_label = self._entity_from_label(label)

        while not finished:
            label, label_confidence = self.most_likely_entity(ent_word_idx, entities)

            confidence = min(confidence, label_confidence)

            if label[2:] != entity_label:
                # words are not tagged the same entity class
                logger.debug(
                    "Inconsistent BILOU tagging found, B- tag, L- "
                    "tag pair encloses multiple entity classes.i.e. "
                    "[B-a, I-b, L-a] instead of [B-a, I-a, L-a].\n"
                    "Assuming B- class is correct."
                )

            if label.startswith("L-"):
                # end of the entity
                finished = True
            elif label.startswith("I-"):
                # middle part of the entity
                ent_word_idx += 1
            else:
                # entity not closed by an L- tag
                finished = True
                ent_word_idx -= 1
                logger.debug(
                    "Inconsistent BILOU tagging found, B- tag not "
                    "closed by L- tag, i.e [B-a, I-a, O] instead of "
                    "[B-a, L-a, O].\nAssuming last tag is L-"
                )
        return ent_word_idx, confidence

    def _handle_bilou_label(self, word_idx, entities):
        label, confidence = self.most_likely_entity(word_idx, entities)
        entity_label = self._entity_from_label(label)

        if self._bilou_from_label(label) == "U":
            return word_idx, confidence, entity_label

        elif self._bilou_from_label(label) == "B":
            # start of multi word-entity need to represent whole extent
            ent_word_idx, confidence = self._find_bilou_end(word_idx, entities)
            return ent_word_idx, confidence, entity_label

        else:
            return None, None, None

    def _from_crf_to_json(
        self, message: Message, entities: List[Any]
    ) -> List[Dict[Text, Any]]:

        if self.pos_features:
            tokens = message.get(MESSAGE_SPACY_FEATURES_NAMES[MESSAGE_TEXT_ATTRIBUTE])
        else:
            tokens = message.get(MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE])

        if len(tokens) != len(entities):
            raise Exception(
                "Inconsistency in amount of tokens between crfsuite and message"
            )

        if self.component_config["BILOU_flag"]:
            return self._convert_bilou_tagging_to_entity_result(
                message, tokens, entities
            )
        else:
            # not using BILOU tagging scheme, multi-word entities are split.
            return self._convert_simple_tagging_to_entity_result(tokens, entities)

    def _convert_bilou_tagging_to_entity_result(
        self, message: Message, tokens: List[Token], entities: List[Dict[Text, float]]
    ):
        # using the BILOU tagging scheme
        json_ents = []
        word_idx = 0
        while word_idx < len(tokens):
            end_idx, confidence, entity_label = self._handle_bilou_label(
                word_idx, entities
            )

            if end_idx is not None:
                ent = self._create_entity_dict(
                    message, tokens, word_idx, end_idx, entity_label, confidence
                )
                json_ents.append(ent)
                word_idx = end_idx + 1
            else:
                word_idx += 1
        return json_ents

    def _convert_simple_tagging_to_entity_result(self, tokens, entities):
        json_ents = []

        for word_idx in range(len(tokens)):
            entity_label, confidence = self.most_likely_entity(word_idx, entities)
            word = tokens[word_idx]
            if entity_label != "O":
                if self.pos_features:
                    start = word.idx
                    end = word.idx + len(word)
                else:
                    start = word.offset
                    end = word.end
                ent = {
                    "start": start,
                    "end": end,
                    "value": word.text,
                    "entity": entity_label,
                    "confidence": confidence,
                }
                json_ents.append(ent)

        return json_ents

    def _sentence_to_features(self, sentence: List[CRFToken]) -> List[Dict[Text, Any]]:
        """Convert a word into discrete features in self.crf_features,
        including word before and word after."""

        configured_features = self.component_config["features"]
        sentence_features = []

        for word_idx in range(len(sentence)):
            # word before(-1), current word(0), next word(+1)
            feature_span = len(configured_features)
            half_span = feature_span // 2
            feature_range = range(-half_span, half_span + 1)
            prefixes = [str(i) for i in feature_range]
            word_features = {}
            for f_i in feature_range:
                if word_idx + f_i >= len(sentence):
                    word_features["EOS"] = True
                    # End Of Sentence
                elif word_idx + f_i < 0:
                    word_features["BOS"] = True
                    # Beginning Of Sentence
                else:
                    word = sentence[word_idx + f_i]
                    f_i_from_zero = f_i + half_span
                    prefix = prefixes[f_i_from_zero]
                    features = configured_features[f_i_from_zero]
                    for feature in features:
                        if feature == "pattern":
                            # add all regexes as a feature
                            regex_patterns = self.function_dict[feature](word)
                            # pytype: disable=attribute-error
                            for p_name, matched in regex_patterns.items():
                                feature_name = prefix + ":" + feature + ":" + p_name
                                word_features[feature_name] = matched
                            # pytype: enable=attribute-error
                        else:
                            # append each feature to a feature vector
                            value = self.function_dict[feature](word)
                            word_features[prefix + ":" + feature] = value
            sentence_features.append(word_features)
        return sentence_features

    @staticmethod
    def _sentence_to_labels(tokens: List[CRFToken],) -> List[Text]:

        return [t.entity for t in tokens]

    def _from_json_to_crf(
        self, message: Message, entity_offsets: List[Tuple[int, int, Text]]
    ) -> List[CRFToken]:
        """Convert json examples to format of underlying crfsuite."""

        if self.pos_features:
            from spacy.gold import GoldParse  # pytype: disable=import-error

            doc_or_tokens = message.get(
                MESSAGE_SPACY_FEATURES_NAMES[MESSAGE_TEXT_ATTRIBUTE]
            )
            gold = GoldParse(doc_or_tokens, entities=entity_offsets)
            ents = [l[5] for l in gold.orig_annot]
        else:
            doc_or_tokens = message.get(MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE])
            ents = self._bilou_tags_from_offsets(doc_or_tokens, entity_offsets)

        # collect badly annotated examples
        collected = []
        for t, e in zip(doc_or_tokens, ents):
            if e == "-":
                collected.append(t)
            elif collected:
                collected_text = " ".join([t.text for t in collected])
                warnings.warn(
                    f"Misaligned entity annotation for '{collected_text}' "
                    f"in sentence '{message.text}' with intent "
                    f"'{message.get('intent')}'. "
                    f"Make sure the start and end values of the "
                    f"annotated training examples end at token "
                    f"boundaries (e.g. don't include trailing "
                    f"whitespaces or punctuation)."
                )
                collected = []

        if not self.component_config["BILOU_flag"]:
            for i, label in enumerate(ents):
                if self._bilou_from_label(label) in {"B", "I", "U", "L"}:
                    # removes BILOU prefix from label
                    ents[i] = self._entity_from_label(label)

        return self._from_text_to_crf(message, ents)

    @staticmethod
    def _bilou_tags_from_offsets(tokens, entities, missing="O"):
        # From spacy.spacy.GoldParse, under MIT License
        starts = {token.offset: i for i, token in enumerate(tokens)}
        ends = {token.end: i for i, token in enumerate(tokens)}
        bilou = ["-" for _ in tokens]
        # Handle entity cases
        for start_char, end_char, label in entities:
            start_token = starts.get(start_char)
            end_token = ends.get(end_char)
            # Only interested if the tokenization is correct
            if start_token is not None and end_token is not None:
                if start_token == end_token:
                    bilou[start_token] = "U-%s" % label
                else:
                    bilou[start_token] = "B-%s" % label
                    for i in range(start_token + 1, end_token):
                        bilou[i] = "I-%s" % label
                    bilou[end_token] = "L-%s" % label
        # Now distinguish the O cases from ones where we miss the tokenization
        entity_chars = set()
        for start_char, end_char, label in entities:
            for i in range(start_char, end_char):
                entity_chars.add(i)
        for n, token in enumerate(tokens):
            for i in range(token.offset, token.end):
                if i in entity_chars:
                    break
            else:
                bilou[n] = missing

        return bilou

    @staticmethod
    def __pattern_of_token(message, i):
        if message.get(MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE]) is not None:
            return message.get(MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE])[i].get(
                "pattern", {}
            )
        else:
            return {}

    @staticmethod
    def __tag_of_token(token):
        if spacy.about.__version__ > "2" and token._.has("tag"):
            return token._.get("tag")
        else:
            return token.tag_

    @staticmethod
    def __get_dense_features(message: Message) -> Optional[List[Any]]:
        features = message.get(
            MESSAGE_VECTOR_DENSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE]
        )

        if features is None:
            return features

        tokens = message.get(MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE], [])
        if len(tokens) != len(features):
            warn_string = f"Number of word embeddings ({len(features)}) does not match number of tokens ({len(tokens)})"
            raise Exception(warn_string)

        # convert to python-crfsuite feature format
        features_out = []
        for feature in features:
            feature_dict = {
                str(index): token_features
                for index, token_features in enumerate(feature)
            }
            converted = {"text_dense_features": feature_dict}
            features_out.append(converted)
        return features_out

    def _from_text_to_crf(
        self, message: Message, entities: List[Text] = None
    ) -> List[CRFToken]:
        """Takes a sentence and switches it to crfsuite format."""

        crf_format = []
        if self.pos_features:
            tokens = message.get(MESSAGE_SPACY_FEATURES_NAMES[MESSAGE_TEXT_ATTRIBUTE])
        else:
            tokens = message.get(MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE])

        text_dense_features = self.__get_dense_features(message)

        for i, token in enumerate(tokens):
            pattern = self.__pattern_of_token(message, i)
            entity = entities[i] if entities else "N/A"
            tag = self.__tag_of_token(token) if self.pos_features else None
            dense_features = (
                text_dense_features[i] if text_dense_features is not None else []
            )

            crf_format.append(
                CRFToken(token.text, tag, entity, pattern, dense_features)
            )

        return crf_format
