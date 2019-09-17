import copy
import datetime
import functools
import logging
import os
import pickle
import warnings

import tensorflow as tf
from typing import Any, Dict, Optional, Text, List

from jsonpickle import json
from six.moves import reduce
import numpy as np

from rasa.nlu.test import determine_token_labels
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData
import rasa.utils.train_utils as train_utils
import rasa.utils.io as io_utils
import rasa.core.utils

from tf_metrics import precision, recall, f1

try:
    import spacy
except ImportError:
    spacy = None

logger = logging.getLogger(__name__)


class ConvLstmCrfEntityExtractor(EntityExtractor):

    provides = ["entities"]

    requires = ["tokens"]

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # dimension of char embeddings
        "dim_chars": 100,
        # dimension of word embeddings
        "dim": 300,
        # dropout
        "dropout": 0.5,
        # number of out of vocabulary buckets
        "num_oov_buckets": 1,
        #
        "buffer": 15000,
        #
        "filters": 50,
        # kernel size of conv
        "kernel_size": 3,
        # batch siye
        "batch_size": 20,
        # number of epochs
        "epochs": 1,
        #
        "lstm_size": 20,
    }
    # end default properties (DOC MARKER - don't remove)

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        session: tf.Session = None,
        graph: tf.Graph = None,
        vocab_words: tf.Tensor = None,
        vocab_chars: tf.Tensor = None,
        vocab_tags: tf.Tensor = None,
        indices: List[int] = None,
        num_tags: int = None,
        num_chars: int = None,
        pred_model: tf.Tensor = None,
    ) -> None:

        self._load_params(component_config)

        self.session = session
        self.graph = graph

        self.vocab_words = vocab_words
        self.vocab_chars = vocab_chars
        self.vocab_tags = vocab_tags
        self.indices = indices
        self.num_tags = num_tags
        self.num_chars = num_chars
        self.pred_model = pred_model

        super(ConvLstmCrfEntityExtractor, self).__init__(component_config)

    def _load_params(self, component_config: Optional[Dict[Text, Any]]) -> None:
        config = copy.deepcopy(self.defaults)
        config.update(component_config)

        self.params = config

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._create_vocab(training_data)

            train_dataset = self._create_dataset(training_data.training_examples)

            iterator = tf.data.Iterator.from_structure(
                train_dataset.output_types,
                train_dataset.output_shapes,
                output_classes=train_dataset.output_classes,
            )
            train_init_op = iterator.make_initializer(train_dataset)

            self.session = tf.Session()

            loss, metrics = self._model(iterator)
            train_op = tf.train.AdamOptimizer().minimize(loss)

            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.local_variables_initializer())
            self.session.run(tf.tables_initializer())

            self._train_model(self.session, train_init_op, train_op, loss, metrics)

    def process(self, message: Message, **kwargs: Any) -> None:
        with self.graph.as_default():
            tf_dataset = self._create_dataset([message], shuffle_and_repeat=False)

            iterator = tf.data.Iterator.from_structure(
                tf_dataset.output_types,
                tf_dataset.output_shapes,
                output_classes=tf_dataset.output_classes,
            )
            pred_init_op = iterator.make_initializer(tf_dataset)

            self.pred_model = self._prediction_model()

            self.session.run(pred_init_op)
            predictions = self.session.run(self.pred_model)
            print (predictions)

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component: Optional["ConvLstmCrfEntityExtractor"] = None,
        **kwargs: Any
    ) -> "ConvLstmCrfEntityExtractor":
        """Loads a policy from the storage.

        **Needs to load its featurizer**
        """

        if not os.path.exists(model_dir):
            raise Exception(
                "Failed to load entity extractor model. Path '{}' "
                "doesn't exist".format(os.path.abspath(model_dir))
            )

        file_name = "tensorflow_embedding.ckpt"
        checkpoint = os.path.join(model_dir, file_name)

        meta_file = os.path.join(model_dir, "conv_lstm_crf_entity_extractor.json")
        params = json.loads(rasa.utils.io.read_file(meta_file))

        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session()
            saver = tf.train.import_meta_graph(checkpoint + ".meta")

            saver.restore(session, checkpoint)

            vocab_words = train_utils.load_tensor("vocab_words")
            vocab_chars = train_utils.load_tensor("vocab_chars")
            vocab_tags = train_utils.load_tensor("vocab_tags")
            pred_model = train_utils.load_tensor("pred_model")

        return cls(
            graph=graph,
            session=session,
            component_config=meta,
            vocab_words=vocab_words,
            vocab_tags=vocab_tags,
            vocab_chars=vocab_chars,
            num_chars=params["num_chars"],
            num_tags=params["num_tags"],
            indices=params["indices"],
            pred_model=pred_model,
        )

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.
        Returns the metadata necessary to load the model again."""

        if self.session is None:
            warnings.warn(
                "Method `persist(...)` was called "
                "without a trained model present. "
                "Nothing to persist then!"
            )
            return

        meta = {
            "indices": self.indices,
            "num_chars": self.num_chars,
            "num_tags": self.num_tags,
        }

        meta_file = os.path.join(model_dir, "conv_lstm_crf_entity_extractor.json")
        rasa.core.utils.dump_obj_as_json_to_file(meta_file, meta)

        file_name = "tensorflow_embedding.ckpt"
        checkpoint = os.path.join(model_dir, file_name)
        io_utils.create_directory_for_file(checkpoint)

        with self.graph.as_default():
            train_utils.persist_tensor("vocab_words", self.vocab_words, self.graph)
            train_utils.persist_tensor("vocab_chars", self.vocab_chars, self.graph)
            train_utils.persist_tensor("vocab_tags", self.vocab_tags, self.graph)
            train_utils.persist_tensor("pred_model", self.pred_model, self.graph)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with open(os.path.join(model_dir, file_name + ".tf_config.pkl"), "wb") as f:
            pickle.dump(self.params, f)

    def _create_vocab(self, train_data: TrainingData):
        words = set()
        tags = set("O")
        chars = set()

        for example in train_data.training_examples:
            words.update([t.text.encode() for t in example.get("tokens", [])])
            chars.update([c.encode() for c in example.text])
            entities = example.get("entities", [])
            for e in entities:
                tags.add(e["entity"].encode())

        self.vocab_tags = tf.contrib.lookup.index_table_from_tensor(list(tags))
        self.reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_tensor(
            list(tags)
        )
        self.vocab_words = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(list(words)), num_oov_buckets=self.params["num_oov_buckets"]
        )
        self.vocab_chars = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(list(chars)), num_oov_buckets=self.params["num_oov_buckets"]
        )
        self.indices = [idx for idx, tag in enumerate(tags) if tag.strip() != "O"]
        self.num_tags = len(self.indices) + 1
        self.num_chars = len(chars) + self.params["num_oov_buckets"]

    def _convert_to_words_tags(self, data: List[Message]):
        for example in data:
            tokens = example.get("tokens", [])
            entities = example.get("entities", [])

            # Encode in Bytes for TF
            words = [w.text.encode() for w in tokens]
            tags = [determine_token_labels(w, entities, None).encode() for w in tokens]
            assert len(words) == len(tags), "Words and tags lengths don't match"

            # Chars
            chars = [[c.encode() for c in w.text] for w in tokens]
            lengths = [len(c) for c in chars]
            max_len = max(lengths)
            chars = [c + [b"<pad>"] * (max_len - l) for c, l in zip(chars, lengths)]
            yield ((words, len(words)), (chars, lengths)), tags

    def _create_dataset(
        self, data: List[Message], shuffle_and_repeat: bool = True, batch: bool = True
    ):
        params = self.params if self.params is not None else {}

        shapes = (
            (
                ([None], ()),  # (words, nwords)
                ([None, None], [None]),
            ),  # (chars, nchars)
            [None],
        )  # tags
        types = (((tf.string, tf.int32), (tf.string, tf.int32)), tf.string)
        defaults = ((("<pad>", 0), ("<pad>", 0)), "O")

        dataset = tf.data.Dataset.from_generator(
            functools.partial(self._convert_to_words_tags, data),
            output_shapes=shapes,
            output_types=types,
        )

        if shuffle_and_repeat:
            dataset = dataset.shuffle(params["buffer"]).repeat(params["epochs"])

        if batch:
            dataset = dataset.padded_batch(
                params.get("batch_size", params["batch_size"]), shapes, defaults
            ).prefetch(1)

        return dataset

    def _model(self, iterator: tf.data.Iterator, training: bool = True):
        dropout = self.params["dropout"]

        # Read vocabs and inputs
        features, labels = iterator.get_next()  # placeholders
        (words, nwords), (chars, nchars) = features

        # Char Embeddings
        char_ids = self.vocab_chars.lookup(chars)
        variable = tf.get_variable(
            "chars_embeddings",
            [self.num_chars + 1, self.params["dim_chars"]],
            tf.float32,
        )
        char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
        char_embeddings = tf.layers.dropout(
            char_embeddings, rate=dropout, training=training
        )

        # Char 1d convolution
        weights = tf.sequence_mask(nchars)
        char_embeddings = self._masked_conv1d_and_max(
            char_embeddings, weights, self.params["filters"], self.params["kernel_size"]
        )

        # Word Embeddings
        word_ids = self.vocab_words.lookup(words)
        glove = np.load(
            "/Users/tabergma/Repositories/tf_ner/data/example/WNUT17/glove.npz"
        )[
            "embeddings"
        ]  # np.array
        variable = np.vstack([glove, [[0.0] * self.params["dim"]]])
        variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
        word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

        # Concatenate Word and Char Embeddings
        embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
        embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

        # LSTM
        t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(self.params["lstm_size"])
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(self.params["lstm_size"])
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
        output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.transpose(output, perm=[1, 0, 2])
        output = tf.layers.dropout(output, rate=dropout, training=training)

        # CRF
        logits = tf.layers.dense(output, self.num_tags)
        crf_params = tf.get_variable(
            "crf", [self.num_tags, self.num_tags], dtype=tf.float32
        )
        self.pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

        # Loss
        tags = self.vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params
        )
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            "acc": tf.metrics.accuracy(tags, self.pred_ids, weights),
            "precision": precision(
                tags, self.pred_ids, self.num_tags, self.indices, weights
            ),
            "recall": recall(tags, self.pred_ids, self.num_tags, self.indices, weights),
            "f1": f1(tags, self.pred_ids, self.num_tags, self.indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        return loss, metrics

    def _prediction_model(self):
        # Predictions
        reverse_vocab_tags = self.reverse_vocab_tags
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(self.pred_ids))
        predictions = {"pred_ids": self.pred_ids, "tags": pred_strings}
        return predictions

    def _masked_conv1d_and_max(self, t, weights, filters, kernel_size):
        """Applies 1d convolution and a masked max-pooling

        Parameters
        ----------
        t : tf.Tensor
            A tensor with at least 3 dimensions [d1, d2, ..., dn-1, dn]
        weights : tf.Tensor of tf.bool
            A Tensor of shape [d1, d2, dn-1]
        filters : int
            number of filters
        kernel_size : int
            kernel size for the temporal convolution

        Returns
        -------
        tf.Tensor
            A tensor of shape [d1, d2, dn-1, filters]

        """
        # Get shape and parameters
        shape = tf.shape(t)
        ndims = t.shape.ndims
        dim1 = reduce(lambda x, y: x * y, [shape[i] for i in range(ndims - 2)])
        dim2 = shape[-2]
        dim3 = t.shape[-1]

        # Reshape weights
        weights = tf.reshape(weights, shape=[dim1, dim2, 1])
        weights = tf.to_float(weights)

        # Reshape input and apply weights
        flat_shape = [dim1, dim2, dim3]
        t = tf.reshape(t, shape=flat_shape)
        t *= weights

        # Apply convolution
        t_conv = tf.layers.conv1d(t, filters, kernel_size, padding="same")
        t_conv *= weights

        # Reduce max -- set to zero if all padded
        t_conv += (1.0 - weights) * tf.reduce_min(t_conv, axis=-2, keepdims=True)
        t_max = tf.reduce_max(t_conv, axis=-2)

        # Reshape the output
        final_shape = [shape[i] for i in range(ndims - 2)] + [filters]
        t_max = tf.reshape(t_max, shape=final_shape)

        return t_max

    def _train_model(self, session, train_init_op, train_op, loss, metrics):

        for epoch in range(self.params["epochs"]):
            print ("-" * 200)
            print ("{} - Starting epoch {}".format(datetime.datetime.now(), epoch))

            session.run(train_init_op)

            batch = 0
            batch_loss = 0
            batch_acc = 0
            batch_f1 = 0
            while True:
                try:
                    _, train_loss, train_metrics = session.run(
                        [train_op, loss, metrics]
                    )
                except tf.errors.OutOfRangeError:
                    break

                if batch % 500 == 0:
                    print (
                        "{} - Processed batch {} - Loss: {}".format(
                            datetime.datetime.now(), batch, train_loss
                        )
                    )
                batch += 1
                batch_loss += train_loss
                batch_acc += train_metrics["acc"][0]
                batch_f1 += train_metrics["f1"][0]

            print (
                "{} - Done processing epoch {} (batches {})".format(
                    datetime.datetime.now(), epoch, batch
                )
            )
            print ("Train loss: {}".format(batch_loss / batch))
            print ("Train accuracy: {}".format(batch_acc / batch))
            print ("Train f1-score: {}".format(batch_f1 / batch))
