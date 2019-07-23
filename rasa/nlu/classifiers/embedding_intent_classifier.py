import io
import logging
import os
import pickle
import typing
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Text, Tuple
from random import shuffle

import numpy as np
from scipy.sparse import issparse, csr_matrix
from tensor2tensor.models.transformer import transformer_small, transformer_prepare_encoder, transformer_encoder
from tensor2tensor.layers import common_attention

from rasa.nlu.classifiers import INTENT_RANKING_LENGTH, NUM_INTENT_CANDIDATES
from rasa.nlu.components import Component
from rasa.utils.common import is_logging_disabled
from sklearn.model_selection import train_test_split
import pandas as pd
from random import shuffle

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from tensorflow import Graph, Session, Tensor
    from rasa.nlu.config import RasaNLUModelConfig
    from rasa.nlu.training_data import TrainingData
    from rasa.nlu.model import Metadata
    from rasa.nlu.training_data import Message

try:
    import tensorflow as tf

    # avoid warning println on contrib import - remove for tf 2
    tf.contrib._warning = None
except ImportError:
    tf = None


class EmbeddingIntentClassifier(Component):
    """Intent classifier using supervised embeddings.

    The embedding intent classifier embeds user inputs
    and intent labels into the same space.
    Supervised embeddings are trained by maximizing similarity between them.
    It also provides rankings of the labels that did not "win".

    The embedding intent classifier needs to be preceded by
    a featurizer in the pipeline.
    This featurizer creates the features used for the embeddings.
    It is recommended to use ``CountVectorsFeaturizer`` that
    can be optionally preceded by ``SpacyNLP`` and ``SpacyTokenizer``.

    Based on the starspace idea from: https://arxiv.org/abs/1709.03856.
    However, in this implementation the `mu` parameter is treated differently
    and additional hidden layers are added together with dropout.
    """

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer for input words
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_a": [256, 128],
        # sizes of hidden layers before the embedding layer for intent labels
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_b": [],

        "share_embedding": False,
        "bidirectional": False,
        "fused_lstm": False,
        "gpu_lstm": False,
        "transformer": False,
        "pos_encoding": "timing",  # {"timing", "emb", "custom_timing"}
        # introduce phase shift in time encodings between transformers
        # 0.5 - 0.8 works on small dataset
        "pos_max_timescale": 1.0e2,
        "max_seq_length": 256,
        "num_heads": 4,
        "use_last": False,

        # training parameters
        "layer_norm": True,
        # initial and final batch sizes - batch size will be
        # linearly increased for each epoch
        "batch_size": [64, 128],
        # "batch_size": 64,
        "stratified_batch": True,
        # number of epochs
        "epochs": 300,

        # embedding parameters
        # dimension size of embedding vectors
        "embed_dim": 20,
        # how similar the algorithm should try
        # to make embedding vectors for correct intent labels
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect intent labels
        "mu_neg": -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
        # the type of the similarity
        "similarity_type": 'cosine',  # string 'cosine' or 'inner'
        "loss_type": 'margin',  # string 'softmax' or 'margin'
        # the number of incorrect intents, the algorithm will minimize
        # their similarity to the input words during training
        "num_neg": 20,

        # include intent sim loss
        "include_intent_sim_loss": True,

        # include text sim loss
        "include_text_sim_loss": True,

        "iou_threshold": 1.0,
        # flag: if true, only minimize the maximum similarity for
        # incorrect intent labels
        "use_max_sim_neg": True,
        # set random seed to any int to get reproducible results
        # try to change to another int if you are not getting good results
        "random_seed": None,

        # regularization parameters
        # the scale of L2 regularization
        "C2": 0.002,
        # the scale of how critical the algorithm should be of minimizing the
        # maximum similarity between embeddings of different intent labels
        "C_emb": 0.8,
        # dropout rate for rnn
        "droprate": 0.0,

        # flag: if true, the algorithm will split the intent labels into tokens
        #       and use bag-of-words representations for them
        "intent_tokenization_flag": False,
        # delimiter string to split the intent labels
        "intent_split_symbol": '_',

        # visualization of accuracy
        # how often to calculate training accuracy
        "evaluate_every_num_epochs": 10,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        "evaluate_on_num_examples": 1000,   # large values may hurt performance

        # Batch size for evaluation runs
        "validation_batch_size": 64


    }

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 inv_intent_dict: Optional[Dict[int, Text]] = None,
                 encoded_all_intents: Optional[np.ndarray] = None,
                 all_intents_embed_values: Optional[np.ndarray] = None,
                 session: Optional['tf.Session'] = None,
                 graph: Optional['tf.Graph'] = None,
                 message_placeholder: Optional['tf.Tensor'] = None,
                 intent_placeholder: Optional['tf.Tensor'] = None,
                 similarity: Optional['tf.Tensor'] = None,
                 all_intents_embed_in: Optional['tf.Tensor'] = None,
                 sim_all: Optional['tf.Tensor'] = None,
                 word_embed: Optional['tf.Tensor'] = None,
                 intent_embed: Optional['tf.Tensor'] = None,
                 ) -> None:
        """Declare instant variables with default values"""

        self._check_tensorflow()
        super(EmbeddingIntentClassifier, self).__init__(component_config)

        self._load_params()

        # transform numbers to intents
        self.inv_intent_dict = inv_intent_dict

        # encode all intents with numbers
        self.encoded_all_intents = encoded_all_intents
        self.all_intents_embed_values = all_intents_embed_values
        self.iou = None

        # tf related instances
        self.session = session
        self.graph = graph
        self.a_in = message_placeholder
        self.b_in = intent_placeholder
        self.sim = similarity

        self.all_intents_embed_in = all_intents_embed_in
        self.sim_all = sim_all

        self.sequence = len(self.a_in.shape) == 3 if self.a_in is not None else None

        # persisted embeddings
        self.word_embed = word_embed
        self.intent_embed = intent_embed

        # Flags to ensure test data is featurized only once
        self.is_test_data_featurized = False


    # init helpers
    def _load_nn_architecture_params(self, config: Dict[Text, Any]) -> None:
        self.hidden_layer_sizes = {'a': config['hidden_layers_sizes_a'],
                                   'b': config['hidden_layers_sizes_b']}

        self.share_embedding = config['share_embedding']
        if self.share_embedding:
            if self.hidden_layer_sizes['a'] != self.hidden_layer_sizes['b']:
                raise ValueError("If embeddings are shared "
                                 "hidden_layer_sizes must coincide")

        self.bidirectional = config['bidirectional']
        self.fused_lstm = config['fused_lstm']
        self.gpu_lstm = config['gpu_lstm']
        self.transformer = config['transformer']
        if (self.gpu_lstm and self.fused_lstm) or (self.transformer and self.fused_lstm) or (self.gpu_lstm and self.transformer):
            raise ValueError("Either `gpu_lstm` or `fused_lstm` or `transformer` should be specified")
        if self.gpu_lstm or self.transformer:
            if any(self.hidden_layer_sizes['a'][0] != size
                   for size in self.hidden_layer_sizes['a']):
                raise ValueError("GPU training only supports identical sizes among layers a")
            if any(self.hidden_layer_sizes['b'][0] != size
                   for size in self.hidden_layer_sizes['b']):
                raise ValueError("GPU training only supports identical sizes among layers b")

        self.pos_encoding = config['pos_encoding']
        self.pos_max_timescale = config['pos_max_timescale']
        self.max_seq_length = config['max_seq_length']
        self.num_heads = config['num_heads']
        self.use_last = config['use_last']

        self.batch_size = config['batch_size']
        self.stratified_batch = config['stratified_batch']
        self.epochs = config['epochs']

    def _load_embedding_params(self, config: Dict[Text, Any]) -> None:
        self.layer_norm = config['layer_norm']
        self.embed_dim = config['embed_dim']
        self.mu_pos = config['mu_pos']
        self.mu_neg = config['mu_neg']
        self.similarity_type = config['similarity_type']
        self.loss_type = config['loss_type']
        self.include_intent_sim_loss = config['include_intent_sim_loss']
        self.include_text_sim_loss = config['include_text_sim_loss']
        self.num_neg = config['num_neg']
        self.iou_threshold = config['iou_threshold']
        self.use_max_sim_neg = config['use_max_sim_neg']
        self.random_seed = self.component_config['random_seed']

    def _load_regularization_params(self, config: Dict[Text, Any]) -> None:
        self.C2 = config['C2']
        self.C_emb = config['C_emb']
        self.droprate = config['droprate']

    def _load_flag_if_tokenize_intents(self, config: Dict[Text, Any]) -> None:
        self.intent_tokenization_flag = config['intent_tokenization_flag']
        self.intent_split_symbol = config['intent_split_symbol']
        if self.intent_tokenization_flag and not self.intent_split_symbol:
            logger.warning("intent_split_symbol was not specified, "
                           "so intent tokenization will be ignored")
            self.intent_tokenization_flag = False

    def _load_visual_params(self, config: Dict[Text, Any]) -> None:
        self.evaluate_every_num_epochs = config['evaluate_every_num_epochs']
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs

        self.evaluate_on_num_examples = config['evaluate_on_num_examples']
        self.validation_bs = config["validation_batch_size"]

    def _load_params(self) -> None:

        self._load_nn_architecture_params(self.component_config)
        self._load_embedding_params(self.component_config)
        self._load_regularization_params(self.component_config)
        self._load_flag_if_tokenize_intents(self.component_config)
        self._load_visual_params(self.component_config)



    # package safety checks
    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow"]

    @staticmethod
    def _check_tensorflow():
        if tf is None:
            raise ImportError(
                'Failed to import `tensorflow`. '
                'Please install `tensorflow`. '
                'For example with `pip install tensorflow`.')

    # training data helpers:
    @staticmethod
    def _create_label_dict(training_data: 'TrainingData', label_type='intent') -> Dict[Text, int]:
        """Create intent dictionary"""

        distinct_intents = set([example.get(label_type)
                                for example in training_data.intent_examples if example.get(label_type) is not None])
        return {intent: idx
                for idx, intent in enumerate(sorted(distinct_intents))}

    @staticmethod
    def _find_example_for_label(intent, examples, label_type="intent"):
        for ex in examples:
            if ex.get(label_type) == intent:
                return ex

    # @staticmethod
    def _create_encoded_labels(self,
                               intent_dict: Dict[Text, int],
                               training_data: 'TrainingData',
                               label_type: Text = "intent",
                               label_feats: Text = "intent_features") -> np.ndarray:
        """Create matrix with intents encoded in rows as bag of words.

        If intent_tokenization_flag is off, returns identity matrix.
        """

        if self.intent_tokenization_flag:
            encoded_all_intents = []

            for key, idx in intent_dict.items():
                encoded_all_intents.insert(
                    idx,
                    self._find_example_for_label(
                        key,
                        training_data.intent_examples,
                        label_type,
                    ).get(label_feats)
                )

            return np.array(encoded_all_intents)
        else:
            return np.eye(len(intent_dict))

    # noinspection PyPep8Naming
    def _prepare_data_for_training(
        self,
        training_data: 'TrainingData',
        intent_dict: Dict[Text, int],
        label_type: Text = "intent",
        non_intents: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""

        filtered_examples = [e for e in training_data.intent_examples]
        if non_intents:
            filtered_examples = [e for e in training_data.intent_examples if not e.get("is_intent")]

        X = np.stack([e.get("text_features") for e in filtered_examples])

        intents_for_X = np.array([intent_dict[e.get(label_type)]
                                  for e in filtered_examples])

        if self.intent_tokenization_flag:
            if non_intents:
                Y = np.stack([e.get("response_features")
                          for e in filtered_examples])
            else:
                Y = np.stack([e.get("intent_features")
                          for e in filtered_examples])

        else:
            Y = np.stack([self.encoded_all_intents[intent_idx]
                          for intent_idx in intents_for_X])

        return X, Y, intents_for_X

    def _create_tf_embed_nn(self, x_in: 'tf.Tensor', is_training: 'tf.Tensor',
                            layer_sizes: List[int], name: Text) -> 'tf.Tensor':
        """Create nn with hidden layers and name"""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        x = x_in
        for i, layer_size in enumerate(layer_sizes):
            x = tf.layers.dense(inputs=x,
                                units=layer_size,
                                activation=tf.nn.relu,
                                kernel_regularizer=reg,
                                name='hidden_layer_{}_{}'.format(name, i),
                                reuse=tf.AUTO_REUSE)
            x = tf.layers.dropout(x, rate=self.droprate, training=is_training)

        return x



    def _create_tf_embed_a(self,
                           a_in: 'tf.Tensor',
                           is_training: 'tf.Tensor',
                           ) -> 'tf.Tensor':
        """Create tf graph for training"""

        a = self._create_tf_embed_nn(a_in, is_training,
                                     self.hidden_layer_sizes['a'],
                                     name='a_and_b' if self.share_embedding else 'a')

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        emb_a = tf.layers.dense(inputs=a,
                                units=self.embed_dim,
                                kernel_regularizer=reg,
                                name='embed_layer_{}'.format('a'),
                                reuse=tf.AUTO_REUSE)

        if self.similarity_type == 'cosine':
            # normalize embedding vectors for cosine similarity
            emb_a = self.normalize_for_cosine(emb_a)

        return emb_a

    def _create_tf_embed_b(self,
                           b_in: 'tf.Tensor',
                           is_training: 'tf.Tensor',
                           ) -> 'tf.Tensor':
        """Create tf graph for training"""


        b = self._create_tf_embed_nn(b_in, is_training,
                                     self.hidden_layer_sizes['b'],
                                     name='a_and_b' if self.share_embedding else 'b')

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        emb_b = tf.layers.dense(inputs=b,
                                units=self.embed_dim,
                                kernel_regularizer=reg,
                                name='embed_layer_{}'.format('b'),
                                reuse=tf.AUTO_REUSE)
        return emb_b


    # training helpers:
    def _linearly_increasing_batch_size(self, epoch: int) -> int:
        """Linearly increase batch size with every epoch.

        The idea comes from https://arxiv.org/abs/1711.00489
        """

        if not isinstance(self.batch_size, list):
            return int(self.batch_size)

        if self.epochs > 1:
            batch_size = int(self.batch_size[0] +
                             epoch * (self.batch_size[1] -
                                      self.batch_size[0]) / (self.epochs - 1))

            return batch_size if batch_size % 2 == 0 else batch_size + 1

        else:
            return int(self.batch_size[0])

    @staticmethod
    def _to_sparse_tensor(array_of_sparse, auto2d=True):
        seq_len = max([x.shape[0] for x in array_of_sparse])
        coo = [x.tocoo() for x in array_of_sparse]
        data = [v for x in array_of_sparse for v in x.data]
        if seq_len == 1 and auto2d:
            indices = [ids for i, x in enumerate(coo) for ids in zip([i] * len(x.row), x.col)]
            return tf.SparseTensor(indices, data, (len(array_of_sparse), array_of_sparse[0].shape[-1]))
        else:
            indices = [ids for i, x in enumerate(coo) for ids in zip([i] * len(x.row), x.row, x.col)]
            return tf.SparseTensor(indices, data, (len(array_of_sparse), seq_len, array_of_sparse[0].shape[-1]))

    @staticmethod
    def _sparse_tensor_to_dense(sparse, units=None, shape=None):
        if shape is None:
            if len(sparse.shape) == 2:
                shape = (tf.shape(sparse)[0], units)
            else:
                shape = (tf.shape(sparse)[0], tf.shape(sparse)[1], units)

        return tf.cast(tf.reshape(tf.sparse_tensor_to_dense(sparse), shape), tf.float32)


    def get_tf_datasets(self, X, Y, dpt_shapes, dpt_types, intents_for_X, batch_size_pl):

        if self.evaluate_on_num_examples:

            X_train, X_val, Y_train, Y_val, X_train_intents, X_val_intents = self.train_val_split(X, Y, intents_for_X)

            train_tf_dataset = None

            if self.stratified_batch:

                if isinstance(self.batch_size, list):
                    train_gen_func = lambda x: self.gen_stratified_batch(X_train, Y_train, X_train_intents, x)
                    train_tf_dataset = self.create_tf_generator_dataset(train_gen_func, dpt_shapes=dpt_shapes,
                                                                        dpt_types=dpt_types, args=([batch_size_pl]))
                else:
                    train_gen_func = lambda: self.gen_stratified_batch(X_train, Y_train, X_train_intents, self.batch_size)
                    train_tf_dataset = self.create_tf_generator_dataset(train_gen_func, dpt_shapes=dpt_shapes,
                                                                        dpt_types=dpt_types, args=([]))

            else:
                if isinstance(self.batch_size, list):
                    train_gen_func = lambda x: self.gen_sequence_batch(X_train, Y_train, x)
                    train_tf_dataset = self.create_tf_generator_dataset(train_gen_func, dpt_shapes=dpt_shapes,
                                                                        dpt_types=dpt_types, args=([batch_size_pl]))
                else:
                    train_gen_func = lambda: self.gen_sequence_batch(X_train, Y_train, self.batch_size)
                    train_tf_dataset = self.create_tf_generator_dataset(train_gen_func, dpt_shapes=dpt_shapes,
                                                                        dpt_types=dpt_types, args=([]))

            val_gen_func = lambda : self.gen_sequence_batch(X_val,Y_val,self.validation_bs)

            val_tf_dataset = self.create_tf_generator_dataset(val_gen_func, dpt_shapes=dpt_shapes, dpt_types=dpt_types,
                                                              args=[])

            return train_tf_dataset, val_tf_dataset

        else:

            train_gen_func = None

            if self.stratified_batch:
                if isinstance(self.batch_size, list):
                    train_gen_func = lambda x: self.gen_stratified_batch(X_train, Y_train, X_train_intents, x)
                else:
                    train_gen_func = lambda: self.gen_stratified_batch(X_train, Y_train, X_train_intents, self.batch_size)


            else:
                if isinstance(self.batch_size, list):
                    train_gen_func = lambda x: self.gen_sequence_batch(X_train, Y_train, x)
                else:
                    train_gen_func = lambda: self.gen_sequence_batch(X_train, Y_train, self.batch_size)

            train_tf_dataset = self.create_tf_generator_dataset(train_gen_func, dpt_shapes=dpt_shapes,
                                                                dpt_types=dpt_types, args=([]))



            return train_tf_dataset, None


    def gen_random_batch(self, X, Y, batch_size):

        num_batches = X.shape[0] // batch_size + int(X.shape[0] % batch_size > 0)
        ind = np.random.choice(np.arange(0, X.shape[0]),size=(num_batches, batch_size))
        for batch_num in range(num_batches):

            batch_ind = ind[batch_num]
            batch_x = X[batch_ind]
            batch_y = Y[batch_ind]

            yield np.array(batch_x), np.array(batch_y)


    def gen_stratified_batch(self, X, Y, intents_for_X, batch_size):

        num_batches = X.shape[0] // batch_size + int(X.shape[0] % batch_size > 0)
        batch_ex_per_intent = max(batch_size//len(set(intents_for_X)), 1)

        # print(X.shape,Y.shape,intents_for_X.shape)
        df = pd.DataFrame({'X': X.tolist(), 'Y': Y.tolist(), 'labels': intents_for_X.tolist()})

        for batch_idx in range(num_batches):

            sampled_df = df.groupby('labels', group_keys=False).apply(lambda x: x.sample(min(X.shape[0], batch_ex_per_intent),replace=True))
            batch_x = sampled_df['X'].tolist()
            batch_y = sampled_df['Y'].tolist()

            yield np.array(batch_x), np.array(batch_y)

    def gen_sequence_batch(self, X, Y, batch_size):

        ids = np.arange(0,X.shape[0])
        X = X[ids]
        Y = Y[ids]

        num_batches = X.shape[0] // batch_size + int(X.shape[0] % batch_size > 0)

        for batch_num in range(num_batches):

            batch_x = X[batch_num * batch_size : (batch_num+1) * batch_size]
            batch_y = Y[batch_num * batch_size : (batch_num+1) * batch_size]

            yield np.array(batch_x), np.array(batch_y)


    def create_tf_generator_dataset(self, gen_func, dpt_types, dpt_shapes, args):

        return tf.data.Dataset.from_generator(gen_func, output_types=dpt_types, output_shapes=dpt_shapes, args=args)


    def get_train_valid_init_op(self, X, Y, intents_for_X, batch_size_in):

        dpt_types = (tf.float32, tf.float32)

        dpt_shapes = ([None, X[0].shape[-1]], [None, Y[0].shape[-1]])

        train_dataset, val_dataset = self.get_tf_datasets(X, Y, dpt_shapes, dpt_types, intents_for_X, batch_size_in)

        # create general iterator
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        batch = iterator.get_next()

        # make datasets that we can initialize separately, but using the same structure via the common iterator
        training_init_op = iterator.make_initializer(train_dataset, name="training_init_op")

        if self.evaluate_on_num_examples:
            validation_init_op = iterator.make_initializer(val_dataset, name="validation_init_op")
            return batch, training_init_op, validation_init_op, iterator
        else:
            return batch, training_init_op, None, iterator

    def train_val_split(self, X, Y, intents_for_X):

        X_train, X_val, Y_train, Y_val, X_train_intents, X_val_intents = train_test_split(X, Y, intents_for_X,
                                                                                          test_size=self.evaluate_on_num_examples,
                                                                                          stratify=intents_for_X)
        return X_train, X_val, Y_train, Y_val, X_train_intents, X_val_intents

    @staticmethod
    def _tf_sample_neg(batch_size: 'tf.Tensor',
                       all_bs: 'tf.Tensor',
                       neg_ids: 'tf.Tensor') -> 'tf.Tensor':
        """Sample negative examples for given indices"""

        tiled_all_bs = tf.tile(tf.expand_dims(all_bs, 0), (batch_size, 1, 1))

        return tf.batch_gather(tiled_all_bs, neg_ids)

    def _tf_calc_iou_mask(self,
                          pos_b: 'tf.Tensor',
                          all_bs: 'tf.Tensor',
                          neg_ids: 'tf.Tensor') -> 'tf.Tensor':
        """Calculate IOU mask for given indices"""

        pos_b_in_flat = tf.expand_dims(pos_b, -2)
        neg_b_in_flat = self._tf_sample_neg(tf.shape(pos_b)[0], all_bs, neg_ids)

        intersection_b_in_flat = tf.minimum(neg_b_in_flat, pos_b_in_flat)
        union_b_in_flat = tf.maximum(neg_b_in_flat, pos_b_in_flat)

        iou = (tf.reduce_sum(intersection_b_in_flat, -1)
               / tf.reduce_sum(union_b_in_flat, -1))
        return 1. - tf.nn.relu(tf.sign(1. - iou))


    def _tf_get_negs(self,
                     all_embed: 'tf.Tensor',
                     all_raw: 'tf.Tensor',
                     raw_pos: 'tf.Tensor') -> Tuple['tf.Tensor', 'tf.Tensor']:
        """Get negative examples from given tensor."""

        batch_size = tf.shape(raw_pos)[0]
        total_cands = tf.shape(all_embed)[0]
        raw_flat = self._tf_make_flat(raw_pos)

        all_indices = tf.tile(tf.expand_dims(tf.range(0, total_cands, 1), 0), (batch_size, 1))
        shuffled_indices = tf.transpose(tf.random.shuffle(tf.transpose(all_indices, (1, 0))), (1, 0))
        neg_ids = tf.slice(shuffled_indices, [0, 0], [-1, tf.math.minimum(total_cands, self.num_neg)])

        # neg_ids = tf.random.categorical(tf.log(tf.ones((batch_size, tf.shape(all_raw)[0]))),
        #                                 self.num_neg)

        bad_negs_flat = self._tf_calc_iou_mask(raw_flat, all_raw, neg_ids)
        bad_negs = tf.reshape(bad_negs_flat, (batch_size, -1))

        neg_embed_flat = self._tf_sample_neg(batch_size, all_embed, neg_ids)
        neg_embed = tf.reshape(neg_embed_flat,
                               (batch_size, -1, all_embed.shape[-1]))

        return neg_embed, bad_negs


    @staticmethod
    def _tf_make_flat(x: 'tf.Tensor') -> 'tf.Tensor':
        """Make tensor 2D."""

        return tf.reshape(x, (-1, x.shape[-1]))

    def _sample_negatives(self, all_intents: 'tf.Tensor') -> Tuple['tf.Tensor',
                                                                   'tf.Tensor',
                                                                   'tf.Tensor',
                                                                   'tf.Tensor',
                                                                   'tf.Tensor',
                                                                   'tf.Tensor']:
        """Sample negative examples."""
        pos_word_embed = tf.expand_dims(self.word_embed, -2)
        neg_word_embed, word_bad_negs = self._tf_get_negs(
            self._tf_make_flat(self.word_embed),
            self._tf_make_flat(self.b_raw),
            self.b_raw
        )
        pos_intent_embed = tf.expand_dims(self.intent_embed, -2)
        neg_intent_embed, intent_bad_negs = self._tf_get_negs(
            self.all_embedded_intents,
            all_intents,
            self.b_raw
        )
        return (pos_word_embed, pos_intent_embed, neg_word_embed, neg_intent_embed,
                word_bad_negs, intent_bad_negs)

    @staticmethod
    def _tf_raw_sim(a: 'tf.Tensor', b: 'tf.Tensor', mask: 'tf.Tensor' = None) -> 'tf.Tensor':
        """Calculate similarity between given tensors."""

        if mask is not None:
            return tf.reduce_sum(a * b, -1) * tf.expand_dims(mask, 2)
        else:
            return tf.reduce_sum(a * b, -1)

    @staticmethod
    def normalize_for_cosine(a):
        return tf.nn.l2_normalize(a, -1)

    def _tf_sim(
            self,
            pos_word_embed: 'tf.Tensor',
            pos_intent_embed: 'tf.Tensor',
            neg_word_embed: 'tf.Tensor',
            neg_intent_embed: 'tf.Tensor',
            word_bad_negs: 'tf.Tensor',
            intent_bad_negs: 'tf.Tensor',
            mask: 'tf.Tensor' = None,
    ) -> Tuple['tf.Tensor', 'tf.Tensor', 'tf.Tensor', 'tf.Tensor', 'tf.Tensor']:
        """Define similarity."""

        # calculate similarity with several
        # embedded actions for the loss
        neg_inf = common_attention.large_compatible_negative(pos_word_embed.dtype)

        sim_pos = self._tf_raw_sim(pos_word_embed, pos_intent_embed, mask)
        sim_neg = self._tf_raw_sim(pos_word_embed, neg_intent_embed,
                                   mask)\
                  + neg_inf * intent_bad_negs
        sim_neg_bot_bot = self._tf_raw_sim(pos_intent_embed, neg_intent_embed,
                                           mask)\
                          + neg_inf * intent_bad_negs
        sim_neg_dial_dial = self._tf_raw_sim(pos_word_embed, neg_word_embed,
                                             mask)\
                            + neg_inf * word_bad_negs
        sim_neg_bot_dial = self._tf_raw_sim(pos_intent_embed, neg_word_embed,
                                            mask)\
                           + neg_inf * word_bad_negs

        # output similarities between user input and bot intents
        # and similarities between bot intents and similarities between user inputs
        return sim_pos, sim_neg, sim_neg_bot_bot, sim_neg_dial_dial, sim_neg_bot_dial

    @staticmethod
    def _tf_calc_accuracy(sim_pos: 'tf.Tensor', sim_neg: 'tf.Tensor') -> 'tf.Tensor':
        """Calculate accuracy"""

        max_all_sim = tf.reduce_max(tf.concat([sim_pos, sim_neg], -1), -1)
        return tf.reduce_mean(tf.cast(tf.math.equal(max_all_sim, sim_pos[:, 0]),
                                      tf.float32))

    def _tf_loss_margin(
        self,
        sim_pos: 'tf.Tensor',
        sim_neg: 'tf.Tensor',
        sim_neg_intent_intent: 'tf.Tensor',
        sim_neg_word_word: 'tf.Tensor',
        sim_neg_intent_word: 'tf.Tensor',
        mask: 'tf.Tensor' = None,
    ) -> 'tf.Tensor':
        """Define max margin loss."""

        # loss for maximizing similarity with correct action
        loss = tf.maximum(0., self.mu_pos - sim_pos[:, 0])

        # loss for minimizing similarity with `num_neg` incorrect actions
        if self.use_max_sim_neg:
            # minimize only maximum similarity over incorrect actions
            max_sim_neg = tf.reduce_max(sim_neg, -1)
            loss += tf.maximum(0., self.mu_neg + max_sim_neg)
        else:
            # minimize all similarities with incorrect actions
            max_margin = tf.maximum(0., self.mu_neg + sim_neg)
            loss += tf.reduce_sum(max_margin, -1)

        # penalize max similarity between pos bot and neg bot embeddings
        max_sim_neg_bot = tf.maximum(0., tf.reduce_max(sim_neg_intent_intent, -1))
        loss += max_sim_neg_bot * self.C_emb

        # penalize max similarity between pos dial and neg dial embeddings
        max_sim_neg_dial = tf.maximum(0., tf.reduce_max(sim_neg_word_word, -1))
        loss += max_sim_neg_dial * self.C_emb

        # penalize max similarity between pos bot and neg dial embeddings
        max_sim_neg_dial = tf.maximum(0., tf.reduce_max(sim_neg_intent_word, -1))
        loss += max_sim_neg_dial * self.C_emb


        # average the loss over sequence length
        if mask is not None:
            # mask loss for different length sequences
            loss *= mask
            loss = tf.reduce_sum(loss, -1) / tf.reduce_sum(mask, 1)
        else:
            loss = tf.reduce_sum(loss, -1)
        # average the loss over the batch
        loss = tf.reduce_mean(loss)

        # add regularization losses
        loss += tf.losses.get_regularization_loss()

        return loss

    @staticmethod
    def _tf_loss_softmax(
        sim_pos: 'tf.Tensor',
        sim_neg: 'tf.Tensor',
        sim_neg_intent_intent: 'tf.Tensor',
        sim_neg_word_word: 'tf.Tensor',
        sim_neg_intent_word: 'tf.Tensor',
        mask: 'tf.Tensor' = None,
    ) -> 'tf.Tensor':
        """Define softmax loss."""

        logits = tf.concat([sim_pos,
                            sim_neg,
                            sim_neg_intent_intent,
                            sim_neg_word_word,
                            sim_neg_intent_word,
                            ], -1)

        # create labels for softmax
        pos_labels = tf.ones_like(logits[:, :1])
        neg_labels = tf.zeros_like(logits[:, 1:])
        labels = tf.concat([pos_labels, neg_labels], -1)

        # mask loss by prediction confidence
        pred = tf.nn.softmax(logits)
        already_learned = tf.pow((1 - pred[:, 0]) / 0.5, 4)

        if mask is not None:

            loss = tf.losses.softmax_cross_entropy(labels,
                                                   logits,
                                                   mask * already_learned)

        else:
            loss = tf.losses.softmax_cross_entropy(labels,
                                                   logits)

        # add regularization losses
        loss += tf.losses.get_regularization_loss()

        return loss

    def _choose_loss(self,
                     sim_pos: 'tf.Tensor',
                     sim_neg: 'tf.Tensor',
                     sim_neg_bot_bot: 'tf.Tensor',
                     sim_neg_dial_dial: 'tf.Tensor',
                     sim_neg_bot_dial: 'tf.Tensor',
                     mask: 'tf.Tensor' = None) -> 'tf.Tensor':
        """Use loss depending on given option."""

        if self.loss_type == 'margin':
            return self._tf_loss_margin(sim_pos, sim_neg,
                                        sim_neg_bot_bot,
                                        sim_neg_dial_dial,
                                        sim_neg_bot_dial,
                                        mask)
        elif self.loss_type == 'softmax':
            return self._tf_loss_softmax(sim_pos, sim_neg,
                                         sim_neg_bot_bot,
                                         sim_neg_dial_dial,
                                         sim_neg_bot_dial,
                                         mask)
        else:
            raise ValueError(
                "Wrong loss type {}, "
                "should be 'margin' or 'softmax'"
                "".format(self.loss_type)
            )


    def build_train_graph(self, X, Y, intents_for_X):

        np.random.seed(self.random_seed)
        tf.set_random_seed(self.random_seed)

        batch_size_in = tf.placeholder(shape=(),dtype=tf.int64)
        is_training = tf.placeholder_with_default(False, shape=())

        (self.a_raw, self.b_raw), train_init_op, val_init_op, iterator = self.get_train_valid_init_op(X, Y, intents_for_X, batch_size_in)

        all_encoded_intents = tf.constant(self.encoded_all_intents, dtype= tf.float32, name="all_intents")

        self.word_embed = self._create_tf_embed_a(self.a_raw, is_training)
        self.intent_embed = self._create_tf_embed_b(self.b_raw, is_training)

        self.all_embedded_intents = self._create_tf_embed_b(all_encoded_intents,is_training)

        (pos_word_embed,
         pos_intent_embed,
         neg_word_embed,
         neg_intent_embed,
         word_bad_negs,
         intent_bad_negs) = self._sample_negatives(all_encoded_intents)

        # calculate similarities
        (sim_pos,
         sim_neg,
         sim_neg_intent_intent,
         sim_neg_word_word,
         sim_neg_intent_word) = self._tf_sim(pos_word_embed,
                                          pos_intent_embed,
                                          neg_word_embed,
                                          neg_intent_embed,
                                          word_bad_negs,
                                          intent_bad_negs)

        self.acc_op = self._tf_calc_accuracy(sim_pos, sim_neg)

        loss = self._choose_loss(sim_pos, sim_neg,
                                 sim_neg_intent_intent,
                                 sim_neg_word_word,
                                 sim_neg_intent_word)

        tf.summary.scalar('total loss', loss)
        tf.summary.scalar('accuracy', self.acc_op)
        train_op = tf.train.AdamOptimizer().minimize(loss)

        self.summary_merged_op = tf.summary.merge_all()

        return batch_size_in, is_training, iterator, loss, train_init_op, train_op, val_init_op


    def build_pred_graph(self, is_training):

        # prediction graph
        # self.all_intents_embed_in = tf.placeholder(tf.float32, (None, None, self.embed_dim),
        #                                            name='all_intents_embed')

        self.a_in = tf.placeholder(self.a_raw.dtype, self.a_raw.shape, name='a')
        self.b_in = tf.placeholder(self.b_raw.dtype, self.b_raw.shape, name='b')

        self.word_embed = self._create_tf_embed_a(self.a_in, is_training)
        # self.intent_embed = self._create_tf_embed_b(self.b_in, is_training)

        self.sim_all = self._tf_raw_sim(
            self.word_embed[:, tf.newaxis, :],
            self.all_embedded_intents[tf.newaxis, :, :],
        )

        if self.similarity_type == "cosine":
            # clip negative values to zero
            confidence = tf.nn.relu(self.sim_all)
        else:
            # normalize result to [0, 1] with softmax
            confidence = tf.nn.softmax(self.sim_all)

        self.intent_embed = self._create_tf_embed_b(self.b_in, is_training)

        self.sim = self._tf_raw_sim(
            self.word_embed[:, tf.newaxis, :],
            self.intent_embed[tf.newaxis, :, :],
        )

        return confidence

    def _train_tf_dataset(self,
                          train_init_op,
                          val_init_op,
                          batch_size_in,
                          loss: 'tf.Tensor',
                          is_training: 'tf.Tensor',
                          train_op: 'tf.Tensor',
                          tb_sum_dir: Text,
                          ) -> None:
        """Train tf graph"""

        self.session.run(tf.global_variables_initializer())

        train_tb_writer = tf.summary.FileWriter(os.path.join(tb_sum_dir,'train'),
                                      self.session.graph)
        test_tb_writer = tf.summary.FileWriter(os.path.join(tb_sum_dir,'test'))

        if self.evaluate_on_num_examples:
            logger.info("Accuracy is updated every {} epochs"
                        "".format(self.evaluate_every_num_epochs))

        pbar = tqdm(range(self.epochs), desc="Epochs", disable=is_logging_disabled())
        train_acc = 0
        val_acc = 0
        train_loss = 0
        val_loss = 0
        interrupted = False

        iter_num = 0
        for ep in pbar:

            batch_size = self._linearly_increasing_batch_size(ep)

            self.session.run(train_init_op, feed_dict={batch_size_in: batch_size})

            ep_train_loss = 0
            ep_train_acc = 0
            batches_per_epoch = 0

            while True:
                try:

                    fetch_list = (self.a_raw, self.b_raw, self.word_embed, self.intent_embed,
                                                                        train_op, loss, self.acc_op,
                                                                        self.summary_merged_op)

                    return_vals = self.session.run(fetch_list, feed_dict={is_training: True, batch_size_in: batch_size})

                    # for i in range(11):
                    #     print(return_vals[i].shape)

                    # print('Sim mat', return_vals[6])
                    # print('Acc', )

                    batch_loss = return_vals[-3]
                    batch_acc = return_vals[-2]

                    train_tb_writer.add_summary(return_vals[-1], iter_num)

                except tf.errors.OutOfRangeError:
                    break
                except KeyboardInterrupt:
                    interrupted = True
                    break

                batches_per_epoch += 1
                iter_num += 1
                ep_train_loss += batch_loss
                ep_train_acc += batch_acc

            # logger.info('Epoch ended with {0} number of mini-batch iterations'.format(batches_per_epoch))

            if interrupted:
                break

            ep_train_loss /= batches_per_epoch
            ep_train_acc /= batches_per_epoch

            if self.evaluate_on_num_examples and val_init_op is not None:

                if (ep == 0 or
                        (ep + 1) % self.evaluate_every_num_epochs == 0 or
                        (ep + 1) == self.epochs):

                    self.session.run(val_init_op)

                    ep_val_loss = 0
                    ep_val_acc = 0
                    val_num_batches = 0

                    while True:

                        try:

                            fetch_list = (self.a_raw, self.b_raw, self.word_embed, self.intent_embed,
                                                                        loss, self.acc_op, self.summary_merged_op)

                            return_vals = self.session.run(fetch_list)

                            # print('Sim mat', return_vals[6])

                            summary = return_vals[-1]
                            batch_acc = return_vals[-2]

                            # print('Acc', batch_acc)

                            batch_loss = return_vals[-3]


                            test_tb_writer.add_summary(summary, iter_num)

                        except tf.errors.OutOfRangeError:
                            break

                        except KeyboardInterrupt:
                            interrupted = True
                            break


                        ep_val_loss += batch_loss
                        ep_val_acc += batch_acc
                        iter_num += 1
                        val_num_batches += 1

                    if interrupted:
                        break

                    ep_val_loss /= val_num_batches
                    ep_val_acc /= val_num_batches

                    pbar.set_postfix({
                        "Train loss": "{:.3f}".format(ep_train_loss),
                        "Train acc": "{:.3f}".format(ep_train_acc),
                        "Val loss": "{:.3f}".format(ep_val_loss),
                        "Val acc": "{:.3f}".format(ep_val_acc)
                    })
            else:
                pbar.set_postfix({
                    "loss": "{:.3f}".format(ep_train_loss)
                })

        # if self.evaluate_on_num_examples:
        #     logger.info("Finished training, "
        #                 "Train loss : {:.3f}, train accuracy: {:.3f}, val loss : {:.3f}, val accuracy: {:.3f}"
        #                 "".format(ep_train_loss, ep_train_acc, ep_val_loss, ep_val_acc))


    def train(self,
              training_data: 'TrainingData',
              cfg: Optional['RasaNLUModelConfig'] = None,
              **kwargs: Any) -> None:
        """Train the embedding intent classifier on a data set."""

        tb_sum_dir = '/tmp/tb_logs/embedding_intent_classifier'

        intent_dict = self._create_label_dict(training_data)
        if len(intent_dict) < 2:
            logger.error("Can not train an intent classifier. "
                         "Need at least 2 different classes. "
                         "Skipping training of intent classifier.")
            return

        self.inv_intent_dict = {v: k for k, v in intent_dict.items()}
        self.encoded_all_intents = self._create_encoded_labels(
            intent_dict, training_data)

        X, Y, intents_for_X = self._prepare_data_for_training(
            training_data, intent_dict)

        if self.share_embedding:
            if X[0].shape[-1] != Y[0].shape[-1]:
                raise ValueError("If embeddings are shared "
                                 "text features and intent features "
                                 "must coincide")

        # check if number of negatives is less than number of intents
        logger.debug("Check if num_neg {} is smaller than "
                     "number of intents {}, "
                     "else set num_neg to the number of intents - 1"
                     "".format(self.num_neg,
                               self.encoded_all_intents.shape[0]))
        self.num_neg = min(self.num_neg,
                           self.encoded_all_intents.shape[0] - 1)

        self.graph = tf.Graph()
        with self.graph.as_default():
            # set random seed
            batch_size_in, is_training, iterator, loss, train_init_op, train_op, val_init_op = \
                self.build_train_graph(X,Y,intents_for_X)
            self.session = tf.Session()

            self._train_tf_dataset(train_init_op, val_init_op, batch_size_in, loss, is_training, train_op, tb_sum_dir)

            self.all_intents_embed_values = self._create_all_intents_embed(self.encoded_all_intents, iterator)

            self.pred_confidence = self.build_pred_graph(is_training)


    def _create_all_intents_embed(self, encoded_all_intents, iterator=None):

        all_intents_embed = []
        batch_size = self._linearly_increasing_batch_size(0)

        if iterator is None:
            batches_per_epoch = (len(encoded_all_intents) // batch_size +
                                 int(len(encoded_all_intents) % batch_size > 0))

            for i in range(batches_per_epoch):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size

                # batch_b = self._to_sparse_tensor(encoded_all_intents[start_idx:end_idx])
                batch_b = self._toarray(encoded_all_intents[start_idx:end_idx])
                # batch_b = encoded_all_intents[start_idx: end_idx]
                # print(batch_size, batches_per_epoch, i, batch_b.shape)

                all_intents_embed.append(self.session.run(self.intent_embed, feed_dict={self.b_raw: batch_b}))
        else:
            if len(iterator.output_shapes[0]) == 2:
                shape_X = (len(encoded_all_intents), iterator.output_shapes[0][-1])
                X_tensor = tf.zeros(shape_X)
            else:
                shape_X = (len(encoded_all_intents), 1, iterator.output_shapes[0][-1])
                X_tensor = tf.zeros(shape_X)

            # X_tensor = tf.SparseTensor(tf.zeros((0, len(iterator.output_shapes[0])), tf.int64),
            #                            tf.zeros((0,), tf.int32), shape_X)
            # Y_tensor = self._to_sparse_tensor(encoded_all_intents)
            Y_tensor = tf.constant(encoded_all_intents, dtype=tf.float32)

            all_intents_dataset = tf.data.Dataset.from_tensor_slices((X_tensor, Y_tensor)).batch(batch_size)
            self.session.run(iterator.make_initializer(all_intents_dataset))

            while True:
                try:
                    all_intents_embed.append(self.session.run(self.intent_embed))
                except tf.errors.OutOfRangeError:
                    break

        all_intents_embed = np.expand_dims(np.concatenate(all_intents_embed, 0), 0)

        return all_intents_embed

    # process helpers
    # noinspection PyPep8Naming
    def _calculate_message_sim(self,
                               X: np.ndarray,
                               all_Y: np.ndarray
                               ) -> Tuple[np.ndarray, List[float]]:
        """Load tf graph and calculate message similarities"""

        message_sim = self.session.run(self.sim_op,
                                       feed_dict={self.a_in: X,
                                                  self.b_in: all_Y})
        message_sim = message_sim.flatten()  # sim is a matrix

        intent_ids = message_sim.argsort()[::-1]
        message_sim[::-1].sort()

        if self.similarity_type == 'cosine':
            # clip negative values to zero
            message_sim[message_sim < 0] = 0
        elif self.similarity_type == 'inner':
            # normalize result to [0, 1] with softmax
            message_sim = np.exp(message_sim)
            message_sim /= np.sum(message_sim)

        # transform sim to python list for JSON serializing
        return intent_ids, message_sim.tolist()

    # noinspection PyPep8Naming
    def _calculate_message_sim_all(self,
                                   X: np.ndarray,
                                   target_intent_id: int = None,
                                   ) -> Tuple[np.ndarray, List[float]]:
        """Load tf graph and calculate message similarities"""

        num_unique_intents = self.all_intents_embed_values.shape[1]

        all_cand_ids = list(range(num_unique_intents))

        if target_intent_id is not None:

            all_cand_ids.pop(target_intent_id)

            filtered_neg_intent_ids = np.random.choice(np.array(all_cand_ids),NUM_INTENT_CANDIDATES-1)

            filtered_cand_ids = np.append(filtered_neg_intent_ids,[target_intent_id])

            filtered_embed_values = self.all_intents_embed_values[:,filtered_cand_ids]

        else:
            filtered_cand_ids = all_cand_ids
            filtered_embed_values = self.all_intents_embed_values

        # print(filtered_cand_ids, filtered_embed_values.shape)

        message_sim = self.session.run(
            self.sim_all,
            feed_dict={self.a_in: X,
                       self.all_intents_embed_in: np.squeeze(filtered_embed_values)}
        )

        # print('Shapes in message sim func', message_sim.shape, X.shape, self.all_intents_embed_values.shape)

        message_sim = message_sim.flatten().tolist()  # sim is a matrix

        # print(message_sim)

        message_sim = list(zip(filtered_cand_ids, message_sim))

        sorted_intents = sorted(message_sim, key=lambda x: x[1])[::-1]

        intent_ids, message_sim = list(zip(*sorted_intents))

        intent_ids = list(intent_ids)
        message_sim = np.array(message_sim)

        # print(intent_ids,message_sim)

        # intent_ids = message_sim.argsort()[::-1]
        # # print(intent_ids)
        # message_sim[::-1].sort()

        if self.similarity_type == 'cosine':
            # clip negative values to zero
            message_sim[message_sim < 0] = 0
        elif self.similarity_type == 'inner':
            # normalize result to [0, 1] with softmax but only over 3*num_neg+1 values
            # message_sim[3*self.num_neg+1:] += -np.inf
            message_sim = np.exp(message_sim)
            message_sim /= np.sum(message_sim)

        # transform sim to python list for JSON serializing
        return np.array(intent_ids), message_sim.tolist()

    def _toarray(self, array_of_sparse):
        if issparse(array_of_sparse):
            return array_of_sparse.toarray()
        elif issparse(array_of_sparse[0]):
            if not self.sequence:
                return np.array([x.toarray() for x in array_of_sparse]).squeeze()
            else:
                seq_len = max([x.shape[0] for x in array_of_sparse])
                X = np.ones([len(array_of_sparse), seq_len, array_of_sparse[0].shape[-1]], dtype=np.int32) * 0
                for i, x in enumerate(array_of_sparse):
                    X[i, :x.shape[0], :] = x.toarray()

                return X
        else:
            return array_of_sparse

    # noinspection PyPep8Naming
    def process(self, message: 'Message', **kwargs: Any) -> None:
        """Return the most likely intent and its similarity to the input."""

        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        if self.session is None:
            logger.error("There is no trained tf.session: "
                         "component is either not trained or "
                         "didn't receive enough training data")

        else:
            # get features (bag of words) for a message
            X = message.get("text_features")
            X = self._toarray(X)

            # Add test intents to existing intents
            if "test_data" in kwargs:

                intent_target = message.get("intent_target")
                test_data = kwargs["test_data"]

                if not self.is_test_data_featurized:

                    logger.info("Embedding test intents and adding to intent list for the first time")

                    new_test_intents = list(set([example.get("intent")
                                                        for example in test_data.intent_examples
                                                        if example.get("intent") not in self.inv_intent_dict.keys()]))

                    self.test_intent_dict = {intent: idx + len(self.inv_intent_dict)
                                                             for idx, intent in enumerate(sorted(new_test_intents))}

                    self.test_inv_intent_dict = {v: k for k, v in self.test_intent_dict.items()}

                    encoded_new_intents = self._create_encoded_labels(self.test_intent_dict, test_data)

                    # self.inv_intent_dict.update(self.test_inv_intent_dict)

                    # Reindex the intents from 0
                    self.test_inv_intent_dict = {i:val for i,(key,val) in enumerate(self.test_inv_intent_dict.items())}
                    self.inv_intent_dict = self.test_inv_intent_dict

                    self.test_intent_dict = {v: k for k, v in self.inv_intent_dict.items()}

                    self.encoded_all_intents = np.append(self.encoded_all_intents, encoded_new_intents, axis=0)

                    new_intents_embed_values = self._create_all_intents_embed(encoded_new_intents)
                    self.all_intents_embed_values = new_intents_embed_values

                    self.is_test_data_featurized = True
                    intent_target_id = self.test_intent_dict[intent_target]

                else:
                    intent_target_id = self.test_intent_dict[intent_target]


            else:
                self.test_intent_dict = self.inv_intent_dict
                intent_target_id = None

            intent_ids, message_sim = self._calculate_message_sim_all(X, intent_target_id)

            # if X contains all zeros do not predict some label
            if X.any() and intent_ids.size > 0:
                intent = {"name": self.inv_intent_dict[intent_ids[0]],
                          "confidence": message_sim[0]}

                ranking = list(zip(list(intent_ids), message_sim))
                ranking = ranking[:INTENT_RANKING_LENGTH]
                intent_ranking = [{"name": self.inv_intent_dict[intent_idx],
                                   "confidence": score}
                                  for intent_idx, score in ranking]

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def _persist_tensor(self, name: Text, tensor: 'tf.Tensor') -> None:
        """Add tensor to collection if it is not None"""

        if tensor is not None:
            self.graph.clear_collection(name)
            self.graph.add_to_collection(name, tensor)

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Dict[Text, Any]:
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

            self._persist_tensor("message_placeholder", self.a_in)
            self._persist_tensor("intent_placeholder", self.b_in)

            self._persist_tensor("similarity_all", self.sim_all)
            self._persist_tensor("similarity", self.sim)

            self._persist_tensor("word_embed", self.word_embed)
            self._persist_tensor("intent_embed", self.intent_embed)
            self._persist_tensor("all_intents_embed", self.all_embedded_intents)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with io.open(os.path.join(
                model_dir,
                file_name + "_inv_intent_dict.pkl"), 'wb') as f:
            pickle.dump(self.inv_intent_dict, f)
        with io.open(os.path.join(
                model_dir,
                file_name + "_encoded_all_intents.pkl"), 'wb') as f:
            pickle.dump(self.encoded_all_intents, f)
        with io.open(os.path.join(
                model_dir,
                file_name + "_all_intents_embed_values.pkl"), 'wb') as f:
            pickle.dump(self.all_intents_embed_values, f)

        return {"file": file_name}

    @staticmethod
    def load_tensor(name: Text) -> Optional['tf.Tensor']:
        """Load tensor or set it to None"""

        tensor_list = tf.get_collection(name)
        return tensor_list[0] if tensor_list else None

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Text = None,
             model_metadata: 'Metadata' = None,
             cached_component: Optional['EmbeddingIntentClassifier'] = None,
             **kwargs: Any
             ) -> 'EmbeddingIntentClassifier':

        if model_dir and meta.get("file"):
            file_name = meta.get("file")
            checkpoint = os.path.join(model_dir, file_name + ".ckpt")

            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session()

                saver = tf.train.import_meta_graph(checkpoint + '.meta')

                saver.restore(sess, checkpoint)

                a_in = cls.load_tensor('message_placeholder')
                b_in = cls.load_tensor('intent_placeholder')

                sim_all = cls.load_tensor("similarity_all")
                sim = cls.load_tensor("similarity")

                word_embed = cls.load_tensor("word_embed")
                intent_embed = cls.load_tensor("intent_embed")
                all_intents_embed = cls.load_tensor("all_intents_embed")

            with io.open(os.path.join(
                    model_dir,
                    file_name + "_inv_intent_dict.pkl"), 'rb') as f:
                inv_intent_dict = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    file_name + "_encoded_all_intents.pkl"), 'rb') as f:
                encoded_all_intents = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    file_name + "_all_intents_embed_values.pkl"), 'rb') as f:
                all_intents_embed_values = pickle.load(f)


            return cls(
                component_config=meta,
                inv_intent_dict=inv_intent_dict,
                encoded_all_intents=encoded_all_intents,
                all_intents_embed_values=all_intents_embed_values,
                session=sess,
                graph=graph,
                message_placeholder=a_in,
                intent_placeholder=b_in,
                similarity=sim,
                all_intents_embed_in=all_intents_embed,
                sim_all=sim_all,
                word_embed=word_embed,
                intent_embed=intent_embed
            )

        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return cls(component_config=meta)
