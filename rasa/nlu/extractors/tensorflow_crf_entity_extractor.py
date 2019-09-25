import copy
import functools
import logging
import os
import pickle
import random
import warnings

import tensorflow as tf
from typing import Any, Dict, Optional, Text, List, Tuple

from jsonpickle import json
from six.moves import reduce
import numpy as np
from tensor2tensor.models.transformer import transformer_small

from rasa.nlu.test import determine_token_labels
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.tokenizers import Token
import rasa.utils.train_utils as train_utils
import rasa.utils.io as io_utils
import rasa.utils.common as common_utils
import rasa.core.utils

from tf_metrics import precision, recall, f1
from tqdm import tqdm

try:
    import spacy
except ImportError:
    spacy = None

logger = logging.getLogger(__name__)


class TensorflowCrfEntityExtractor(EntityExtractor):

    provides = ["entities"]

    requires = ["tokens"]

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # file to word embeddings in numpy format
        "word_embeddings_file": "/Users/tabergma/Repositories/tf_ner/data/example/WNUT17/glove.npz",
        # dimension of char embeddings
        "dim_chars": 100,
        # dimension of word embeddings
        "dim": 300,
        # dropout
        "dropout": 0.5,
        # number of out of vocabulary buckets
        "num_oov_buckets": 1,
        # see https://www.tensorflow.org/api_docs/python/tf/compat/v2/data/Dataset#shuffle
        "buffer": 15000,
        # the number of filters in the convolution
        "filters": 50,
        # kernel size of conv
        "kernel_size": 3,
        # batch siye
        "batch_size": 20,
        # number of epochs
        "epochs": 25,
        # the number of units in the LSTM cell
        "lstm_size": 20,
        # how often calculate validation metrics
        "evaluate_every_num_epochs": 5,  # small values may hurt performance
        # how many examples to use for hold out validation set
        "evaluate_on_num_examples": 0,  # large values may hurt performance
        "use_transformer": False,
        "C2": 0.002,
        "layer_sizes": [256],
        "num_heads": 4,
        "pos_encoding": "timing",  # {"timing", "emb", "custom_timing"}
        "max_seq_length": 256,
        "pos_max_timescale": 1.0e2,
        "use_last": False,
        "bidirectional": True,
    }
    # end default properties (DOC MARKER - don't remove)

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        session: Optional[tf.Session] = None,
        graph: Optional[tf.Graph] = None,
        indices: Optional[List[int]] = None,
        num_tags: Optional[int] = None,
        num_chars: Optional[int] = None,
        chars: Optional[tf.placeholder] = None,
        nchars: Optional[tf.placeholder] = None,
        words: Optional[tf.placeholder] = None,
        nwords: Optional[tf.placeholder] = None,
        predictions: Optional[tf.Tensor] = None,
    ) -> None:

        self._load_params(component_config)

        self.session = session
        self.graph = graph

        # vocabulary - created during training
        self.indices = indices
        self.num_tags = num_tags
        self.num_chars = num_chars

        # lookup tables
        self.vocab_tags = None
        self.reverse_vocab_tags = None
        self.vocab_words = None
        self.vocab_chars = None

        # feautes
        self.chars = chars
        self.nchars = nchars
        self.words = words
        self.nwords = nwords

        # input for predictions
        self.predictions = predictions

        super(TensorflowCrfEntityExtractor, self).__init__(component_config)

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

            train_dataset, eval_dataset = self._create_datasets(
                training_data.training_examples
            )
            iterator = tf.data.Iterator.from_structure(
                train_dataset.output_types,
                train_dataset.output_shapes,
                output_classes=train_dataset.output_classes,
            )
            train_init_op = iterator.make_initializer(train_dataset)

            if eval_dataset:
                eval_init_op = iterator.make_initializer(eval_dataset)
            else:
                eval_init_op = None

            self.session = tf.Session()

            loss, metrics = self._build_train_graph(iterator)
            train_op = tf.train.AdamOptimizer().minimize(loss)

            self._build_prediction_graph()

            self._train_model(train_init_op, eval_init_op, train_op, loss, metrics)

    def process(self, message: Message, **kwargs: Any) -> None:
        with self.graph.as_default():
            features, _ = self._convert_message(message)
            (words, nwords), (chars, nchars) = features

            output = self.session.run(
                self.predictions,
                feed_dict={
                    self.words: [words],
                    self.nwords: [nwords],
                    self.chars: [chars],
                    self.nchars: [nchars],
                },
            )

            # Predictions
            tags = output[0]
            entities = self._convert_tags_to_entities(
                message.text, message.get("tokens", []), tags
            )

            extracted = self.add_extractor_name(entities)
            message.set(
                "entities", message.get("entities", []) + extracted, add_to_output=True
            )

    def _convert_tags_to_entities(
        self, text: str, tokens: List[Token], tags: List[bytes]
    ) -> List[Dict[Text, Any]]:
        entities = []
        last_tag = "O"
        for token, tag in zip(tokens, tags):
            tag = tag.decode("utf-8")
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

    def _convert_message(
        self, data: Message
    ) -> Tuple[
        Tuple[Tuple[List[Text], int], Tuple[List[List[bytes]], List[int]]], List[bytes]
    ]:
        tokens = data.get("tokens", [])
        entities = data.get("entities", [])

        # Encode in Bytes for TF
        words = [w.text.encode() for w in tokens]
        tags = [determine_token_labels(w, entities, None).encode() for w in tokens]
        assert len(words) == len(tags), "Words and tags lengths don't match"

        # Chars
        chars = [[c.encode() for c in w.text] for w in tokens]
        lengths = [len(c) for c in chars]
        max_len = max(lengths)
        chars = [c + [b"<pad>"] * (max_len - l) for c, l in zip(chars, lengths)]
        return ((words, len(words)), (chars, lengths)), tags

    def _convert_messages(self, data: List[Message]):
        for example in data:
            yield self._convert_message(example)

    def _split_data(self, data):
        eval_count = self.params["evaluate_on_num_examples"]
        if eval_count > 0:
            random.shuffle(data)
            return data[0:eval_count], data[eval_count:]
        return [], data

    def _create_datasets(
        self, data: List[Message], shuffle_and_repeat: bool = True
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
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

        eval_data, train_data = self._split_data(data)

        train_dataset = self._create_dataset(
            data, defaults, params, shapes, shuffle_and_repeat, types
        )

        if eval_data:
            eval_dataset = self._create_dataset(
                data, defaults, params, shapes, shuffle_and_repeat, types
            )
        else:
            eval_dataset = None

        return train_dataset, eval_dataset

    def _create_dataset(
        self, data, defaults, params, shapes, shuffle_and_repeat, types
    ):
        dataset = tf.data.Dataset.from_generator(
            functools.partial(self._convert_messages, data),
            output_shapes=shapes,
            output_types=types,
        )

        if shuffle_and_repeat:
            dataset = dataset.shuffle(params["buffer"]).repeat(params["epochs"])

        return dataset.padded_batch(
            params.get("batch_size", 20), shapes, defaults
        ).prefetch(1)

    def _build_train_graph(
        self, iterator: tf.data.Iterator, training: bool = True
    ) -> Tuple[float, Dict[Text, Any]]:
        # tf.enable_eager_execution()
        dropout = self.params["dropout"]

        # labels = Tensor(seq, batch)
        features, labels = iterator.get_next()
        # (Tensor(seq, batch), Tensor(batch)), (Tensor(seq, seq, batch), Tensor(seq, batch))
        (self.words, self.nwords), (self.chars, self.nchars) = features

        # Tensor(seq, batch, dim)
        embeddings = self._featurization(dropout, training)

        if self.params["use_transformer"]:
            # Transformer
            x = self._create_transformer(embeddings, training)
        else:
            # LSTM
            # Tensor(seq, batch, lstm-size * 2) e.g. bidirectional
            x = self._create_lstm(embeddings, dropout, training)

        # CRF
        logits, crf_params, pred_ids = self._create_crf(x)

        # Loss
        tags = self.vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, self.nwords, crf_params
        )
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(self.nwords)
        metrics = {
            "acc": tf.metrics.accuracy(tags, pred_ids, weights),
            "precision": precision(
                tags, pred_ids, self.num_tags, self.indices, weights
            ),
            "recall": recall(tags, pred_ids, self.num_tags, self.indices, weights),
            "f1": f1(tags, pred_ids, self.num_tags, self.indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        return loss, metrics

    def _build_prediction_graph(self):
        self.words = tf.placeholder(dtype=tf.string, shape=(None, None))
        self.nwords = tf.placeholder(dtype=tf.int32, shape=(None))
        self.chars = tf.placeholder(dtype=tf.string, shape=(None, None, None))
        self.nchars = tf.placeholder(dtype=tf.int32, shape=(None, None))

        # Read vocabs and inputs
        embeddings = self._featurization(training=False)

        if self.params["use_transformer"]:
            # Transformer
            x = self._create_transformer(embeddings, training=False)
        else:
            # LSTM
            # Tensor(seq, batch, lstm-size * 2) e.g. bidirectional
            x = self._create_lstm(embeddings, training=False)

        # CRF
        _, _, pred_ids = self._create_crf(x)

        # get predictions
        self.predictions = self.reverse_vocab_tags.lookup(tf.to_int64(pred_ids))

    def _create_crf(
        self, output_lstm: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(output_lstm, self.num_tags, name="crf-logits")
            crf_params = tf.get_variable(
                "crf-params", [self.num_tags, self.num_tags], dtype=tf.float32
            )
            pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, self.nwords)
            return logits, crf_params, pred_ids

    def _create_transformer(self, embeddings: tf.Tensor, training: bool = True):
        # mask different length sequences
        mask = tf.sign(tf.reduce_max(embeddings, -1))
        last = mask * tf.cumprod(1 - mask, axis=1, exclusive=True, reverse=True)
        mask = tf.cumsum(last, axis=1, reverse=True)

        hparams = transformer_small()

        hparams.num_hidden_layers = len(self.params["layer_sizes"])
        hparams.hidden_size = self.params["layer_sizes"][0]
        # it seems to be factor of 4 for transformer architectures in t2t
        hparams.filter_size = self.params["layer_sizes"][0] * 4
        hparams.num_heads = self.params["num_heads"]
        hparams.pos = self.params["pos_encoding"]

        hparams.max_length = self.params["max_seq_length"]
        if not self.params["bidirectional"]:
            hparams.unidirectional_encoder = True

        # When not in training mode, set all forms of dropout to zero.
        for key, value in hparams.values().items():
            if key.endswith("dropout") or key == "label_smoothing":
                setattr(hparams, key, value * tf.cast(training, tf.float32))

        attention_weights = {}
        x = train_utils.create_t2t_transformer_encoder(
            embeddings,
            mask,
            attention_weights,
            hparams,
            self.params["C2"],
            tf.convert_to_tensor(training),
        )

        return x

    def _create_lstm(
        self,
        embeddings: tf.Tensor,
        dropout: Optional[float] = None,
        training: bool = True,
    ) -> tf.Tensor:
        t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(
            self.params["lstm_size"], reuse=tf.AUTO_REUSE
        )
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(
            self.params["lstm_size"], reuse=tf.AUTO_REUSE
        )
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=self.nwords)
        output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=self.nwords)
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.transpose(output, perm=[1, 0, 2])
        output = tf.layers.dropout(output, rate=dropout, training=training)
        return output

    def _featurization(
        self, dropout: Optional[float] = None, training: bool = True
    ) -> tf.Tensor:
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            # Char Embeddings
            char_ids = self.vocab_chars.lookup(self.chars)
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
            weights = tf.sequence_mask(self.nchars)
            char_embeddings = self._masked_conv1d_and_max(
                char_embeddings,
                weights,
                self.params["filters"],
                self.params["kernel_size"],
            )

            # Word Embeddings
            word_ids = self.vocab_words.lookup(self.words)
            loaded_word_embeddings = np.load(self.params["word_embeddings_file"])[
                "embeddings"
            ]  # np.array
            variable = np.vstack([loaded_word_embeddings, [[0.0] * self.params["dim"]]])
            variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
            word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

            # Concatenate Word and Char Embeddings
            embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
            embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

            return embeddings

    def _masked_conv1d_and_max(
        self, tensor: tf.Tensor, weights: tf.Tensor, filters: int, kernel_size: int
    ) -> tf.Tensor:
        """
        Applies 1d convolution and a masked max-pooling.

        Args:
            tensor: A tensor with at least 3 dimensions [d1, d2, ..., dn-1, dn]
            weights: A Tensor of shape [d1, d2, dn-1]
            filters: number of filters
            kernel_size: kernel size for the temporal convolution

        Returns: A tensor of shape [d1, d2, dn-1, filters]
        """
        # Get shape and parameters
        shape = tf.shape(tensor)
        ndims = tensor.shape.ndims
        dim1 = reduce(lambda x, y: x * y, [shape[i] for i in range(ndims - 2)])
        dim2 = shape[-2]
        dim3 = tensor.shape[-1]

        # Reshape weights
        weights = tf.reshape(weights, shape=[dim1, dim2, 1])
        weights = tf.to_float(weights)

        # Reshape input and apply weights
        flat_shape = [dim1, dim2, dim3]
        tensor = tf.reshape(tensor, shape=flat_shape)
        tensor *= weights

        # Apply convolution
        t_conv = tf.layers.conv1d(tensor, filters, kernel_size, padding="same")
        t_conv *= weights

        # Reduce max -- set to zero if all padded
        t_conv += (1.0 - weights) * tf.reduce_min(t_conv, axis=-2, keepdims=True)
        t_max = tf.reduce_max(t_conv, axis=-2)

        # Reshape the output
        final_shape = [shape[i] for i in range(ndims - 2)] + [filters]
        t_max = tf.reshape(t_max, shape=final_shape)

        return t_max

    def _train_model(
        self,
        train_init_op: tf.Tensor,
        eval_init_op: tf.Tensor,
        train_op: tf.Tensor,
        loss: float,
        metrics: Dict[Text, Any],
    ):
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
        self.session.run(tf.tables_initializer())

        pbar = tqdm(
            range(self.params["epochs"]),
            desc="Epochs",
            disable=common_utils.is_logging_disabled(),
        )

        train_loss = 0
        train_acc = 0
        eval_loss = 0
        eval_acc = 0

        for epoch in pbar:
            train_acc, train_f1, train_loss = self._process_data(
                train_init_op, train_op, loss, metrics
            )

            postfix_dict = {
                "loss": "{:.3f}".format(train_loss),
                "acc": "{:.3f}".format(train_acc),
                "f1": "{:.3f}".format(train_f1),
            }

            if eval_init_op is not None:
                if (epoch + 1) % self.params["evaluate_every_num_epochs"] == 0 or (
                    epoch + 1
                ) == self.params["epochs"]:

                    eval_acc, eval_f1, eval_loss = self._process_data(
                        eval_init_op, None, loss, metrics
                    )

                    postfix_dict.update(
                        {
                            "val_loss": "{:.3f}".format(eval_loss),
                            "val_acc": "{:.3f}".format(eval_acc),
                            "val_f1": "{:.3f}".format(eval_f1),
                        }
                    )

            pbar.set_postfix(postfix_dict)

        final_message = (
            "Finished training TensorflowCrfEntityExtractor, "
            "train loss={:.3f}, train accuracy={:.3f}"
            "".format(train_loss, train_acc)
        )
        if eval_init_op is not None:
            final_message += (
                ", validation loss={:.3f}, validation accuracy={:.3f}"
                "".format(eval_loss, eval_acc)
            )
        logger.info(final_message)

    def _process_data(
        self,
        init_op: tf.Tensor,
        init_run: Optional[tf.Tensor],
        loss: float,
        metrics: Dict[Text, Any],
    ):
        steps = 0
        batch_loss = 0
        batch_acc = 0
        batch_f1 = 0

        self.session.run(init_op)

        while True:
            try:
                if init_run:
                    _, _loss, _metrics = self.session.run([init_run, loss, metrics])
                else:
                    _loss, _metrics = self.session.run([loss, metrics])
            except tf.errors.OutOfRangeError:
                break

            steps += 1
            batch_loss += _loss
            batch_acc += _metrics["acc"][0]
            batch_f1 += _metrics["f1"][0]

        batch_loss = batch_loss / steps
        batch_acc = batch_acc / steps
        batch_f1 = batch_f1 / steps

        return batch_acc, batch_f1, batch_loss

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component: Optional["TensorflowCrfEntityExtractor"] = None,
        **kwargs: Any
    ) -> "TensorflowCrfEntityExtractor":
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

            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())

            saver = tf.train.import_meta_graph(checkpoint + ".meta")

            session.run(tf.tables_initializer())

            saver.restore(session, checkpoint)

            chars = train_utils.load_tensor("chars")
            nchars = train_utils.load_tensor("nchars")
            words = train_utils.load_tensor("words")
            nwords = train_utils.load_tensor("nwords")
            predictions = train_utils.load_tensor("predictions")

        return cls(
            graph=graph,
            session=session,
            component_config=meta,
            num_chars=params["num_chars"],
            num_tags=params["num_tags"],
            indices=params["indices"],
            words=words,
            nwords=nwords,
            chars=chars,
            nchars=nchars,
            predictions=predictions,
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
            train_utils.persist_tensor("chars", self.chars, self.graph)
            train_utils.persist_tensor("nchars", self.nchars, self.graph)
            train_utils.persist_tensor("words", self.words, self.graph)
            train_utils.persist_tensor("nwords", self.nwords, self.graph)
            train_utils.persist_tensor("predictions", self.predictions, self.graph)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with open(os.path.join(model_dir, file_name + ".tf_config.pkl"), "wb") as f:
            pickle.dump(self.params, f)
