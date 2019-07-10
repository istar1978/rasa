import io
import logging
import numpy as np
import os
import pickle
import typing
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa.nlu.classifiers import INTENT_RANKING_LENGTH
from rasa.nlu.components import Component
from rasa.utils.common import is_logging_disabled
from rasa.nlu.classifiers.embedding_intent_classifier import EmbeddingIntentClassifier

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

class ResponseSelector(EmbeddingIntentClassifier):
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

    provides = ["response", "response_ranking"]

    requires = ["text_features"]

    name = 'response_selector'

    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer for input words
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_a": [256, 128],
        # sizes of hidden layers before the embedding layer for intent labels
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_b": [],

        "response_type": "chitchat",

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
        "droprate": 0.2,

        # flag: if true, the algorithm will split the intent labels into tokens
        #       and use bag-of-words representations for them
        "intent_tokenization_flag": False,
        # delimiter string to split the intent labels
        "intent_split_symbol": '_',

        # visualization of accuracy
        # how often to calculate training accuracy
        "evaluate_every_num_epochs": 10,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        "evaluate_on_num_examples": 1000,  # large values may hurt performance

        # Batch size for evaluation runs
        "validation_batch_size": 64

    }

    def __init__(
            self,
            component_config: Optional[Dict[Text, Any]] = None,
            inv_response_dict: Optional[Dict[int, Text]] = None,
            encoded_all_responses: Optional[np.ndarray] = None,
            all_responses_embed_values: Optional[np.ndarray] = None,
            session: Optional['tf.Session'] = None,
            graph: Optional['tf.Graph'] = None,
            message_placeholder: Optional['tf.Tensor'] = None,
            response_placeholder: Optional['tf.Tensor'] = None,
            similarity_op: Optional['tf.Tensor'] = None,
            all_responses_embed_in: Optional['tf.Tensor'] = None,
            sim_all: Optional['tf.Tensor'] = None,
            word_embed: Optional['tf.Tensor'] = None,
            response_embed: Optional['tf.Tensor'] = None,
    ) -> None:
        super(ResponseSelector, self).__init__(component_config, inv_response_dict, encoded_all_responses, all_responses_embed_values,
                                               session, graph, message_placeholder, response_placeholder,
                                               similarity_op, all_responses_embed_in, sim_all, word_embed, response_embed)

    # noinspection PyPep8Naming
    def process(self, message: 'Message', **kwargs: Any) -> None:
        """Return the most likely intent and its similarity to the input."""

        response = {"name": None, "confidence": 0.0}
        response_ranking = []

        if self.session is None:
            logger.error("There is no trained tf.session: "
                         "component is either not trained or "
                         "didn't receive enough training data")

        else:
            # get features (bag of words) for a message
            X = message.get("text_features")
            X = self._toarray(X)

            # Add test responses to existing responses
            if "test_data" in kwargs:

                response_target = message.get("response_target")

                if message.get("response_target") is None:
                    message.set("response", response, add_to_output=True)
                    message.set("response_ranking", response_ranking, add_to_output=True)
                    return

                test_data = kwargs["test_data"]

                if not self.is_test_data_featurized:

                    logger.info("Embedding test responses and adding to response list for the first time")

                    new_test_intents = list(set([example.get("response")
                                                 for example in test_data.intent_examples
                                                 if example.get("response") is not None and example.get("response") not in self.inv_intent_dict.keys()]))

                    self.test_intent_dict = {intent: idx + len(self.inv_intent_dict)
                                             for idx, intent in enumerate(sorted(new_test_intents))}

                    self.test_inv_intent_dict = {v: k for k, v in self.test_intent_dict.items()}

                    encoded_new_intents = self._create_encoded_labels(self.test_intent_dict, test_data)

                    # Reindex the intents from 0
                    self.test_inv_intent_dict = {i: val for i, (key, val) in
                                                 enumerate(self.test_inv_intent_dict.items())}

                    self.inv_intent_dict = self.test_inv_intent_dict

                    self.test_intent_dict = {v: k for k, v in self.inv_intent_dict.items()}

                    self.encoded_all_intents = np.append(self.encoded_all_intents, encoded_new_intents, axis=0)

                    new_intents_embed_values = self._create_all_intents_embed(encoded_new_intents)
                    self.all_intents_embed_values = new_intents_embed_values

                    self.is_test_data_featurized = True
                    intent_target_id = self.test_intent_dict[response_target]

                else:
                    intent_target_id = self.test_intent_dict[response_target]

            else:
                self.test_intent_dict = self.inv_intent_dict
                intent_target_id = None

            intent_ids, message_sim = self._calculate_message_sim_all(X, intent_target_id)

            # if X contains all zeros do not predict some label
            if X.any() and intent_ids.size > 0:
                response = {"name": self.inv_intent_dict[intent_ids[0]],
                          "confidence": message_sim[0]}

                ranking = list(zip(list(intent_ids), message_sim))
                ranking = ranking[:INTENT_RANKING_LENGTH]
                response_ranking = [{"name": self.inv_intent_dict[intent_idx],
                                   "confidence": score}
                                  for intent_idx, score in ranking]

        message.set("response", response, add_to_output=True)
        message.set("response_ranking", response_ranking, add_to_output=True)


    def train(self,
              training_data: 'TrainingData',
              cfg: Optional['RasaNLUModelConfig'] = None,
              **kwargs: Any) -> None:
        """Train the embedding intent classifier on a data set."""

        tb_sum_dir = '/tmp/tb_logs/response_selector'

        intent_dict = self._create_label_dict(training_data,label_type='response')

        if len(intent_dict) < 2:
            logger.error("Can not train a response selector. "
                         "Need at least 2 different classes. "
                         "Skipping training of response selector.")
            return

        self.inv_intent_dict = {v: k for k, v in intent_dict.items()}
        self.encoded_all_intents = self._create_encoded_labels(
            intent_dict, training_data, label_type="response", label_feats="response_features")

        X, Y, intents_for_X = self._prepare_data_for_training(
            training_data, intent_dict, label_type='response', non_intents=True)

        if self.share_embedding:
            if X[0].shape[-1] != Y[0].shape[-1]:
                raise ValueError("If embeddings are shared "
                                 "text features and intent features "
                                 "must coincide")

        # check if number of negatives is less than number of intents
        logger.debug("Check if num_neg {} is smaller than "
                     "number of response {}, "
                     "else set num_neg to the number of responses - 1"
                     "".format(self.num_neg,
                               self.encoded_all_intents.shape[0]))
        self.num_neg = min(self.num_neg,
                           self.encoded_all_intents.shape[0] - 1)

        self.graph = tf.Graph()
        with self.graph.as_default():
            # set random seed
            batch_size_in, is_training, iterator, loss, train_init_op, train_op, val_init_op = self.build_train_graph(X,
                                                                                                                      Y)
            self.session = tf.Session()

            self._train_tf_dataset(train_init_op, val_init_op, batch_size_in, loss, is_training, train_op, tb_sum_dir)

            self.all_intents_embed_values = self._create_all_intents_embed(self.encoded_all_intents, iterator)

            self.build_pred_graph(is_training)


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
            if not self.gpu_lstm:
                self.graph.clear_collection('message_placeholder')
                self.graph.add_to_collection('message_placeholder',
                                             self.a_in)

                self.graph.clear_collection('intent_placeholder')
                self.graph.add_to_collection('intent_placeholder',
                                             self.b_in)

                self.graph.clear_collection('similarity_op')
                self.graph.add_to_collection('similarity_op',
                                             self.sim_op)

                self.graph.clear_collection('all_intents_embed_in')
                self.graph.add_to_collection('all_intents_embed_in',
                                             self.all_intents_embed_in)
                self.graph.clear_collection('sim_all')
                self.graph.add_to_collection('sim_all',
                                             self.sim_all)

                self.graph.clear_collection('word_embed')
                self.graph.add_to_collection('word_embed',
                                             self.word_embed)
                self.graph.clear_collection('intent_embed')
                self.graph.add_to_collection('intent_embed',
                                             self.intent_embed)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        placeholder_dims = {'a_in': np.int(self.a_in.shape[-1]),
                            'b_in': np.int(self.b_in.shape[-1])}
        with io.open(os.path.join(
                model_dir,
                file_name + "_placeholder_dims.pkl"), 'wb') as f:
            pickle.dump(placeholder_dims, f)
        with io.open(os.path.join(
                model_dir,
                file_name + "_inv_responses_dict.pkl"), 'wb') as f:
            pickle.dump(self.inv_intent_dict, f)
        with io.open(os.path.join(
                model_dir,
                file_name + "_encoded_all_responses.pkl"), 'wb') as f:
            pickle.dump(self.encoded_all_intents, f)
        with io.open(os.path.join(
                model_dir,
                file_name + "_all_responses_embed_values.pkl"), 'wb') as f:
            pickle.dump(self.all_intents_embed_values, f)

        return {"file": file_name}

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
                if meta['gpu_lstm']:
                    # rebuild tf graph for prediction
                    with io.open(os.path.join(
                            model_dir,
                            file_name + "_placeholder_dims.pkl"), 'rb') as f:
                        placeholder_dims = pickle.load(f)
                    reg = tf.contrib.layers.l2_regularizer(meta['C2'])

                    a_in = tf.placeholder(tf.float32, (None, None, placeholder_dims['a_in']),
                                          name='a')
                    b_in = tf.placeholder(tf.float32, (None, None, placeholder_dims['b_in']),
                                          name='b')
                    a = cls._create_tf_gpu_predict_embed(meta, a_in,
                                                         meta['hidden_layers_sizes_a'],
                                                         name='a_and_b' if meta['share_embedding'] else 'a')
                    word_embed = tf.layers.dense(inputs=a,
                                                 units=meta['embed_dim'],
                                                 kernel_regularizer=reg,
                                                 name='embed_layer_{}'.format('a'),
                                                 reuse=tf.AUTO_REUSE)

                    b = cls._create_tf_gpu_predict_embed(meta, b_in,
                                                         meta['hidden_layers_sizes_b'],
                                                         name='a_and_b' if meta['share_embedding'] else 'b')
                    intent_embed = tf.layers.dense(inputs=b,
                                                   units=meta['embed_dim'],
                                                   kernel_regularizer=reg,
                                                   name='embed_layer_{}'.format('b'),
                                                   reuse=tf.AUTO_REUSE)

                    tiled_intent_embed = cls._tf_sample_neg(intent_embed, None, None,
                                                            tf.shape(word_embed)[0])

                    sim_op = cls._tf_gpu_sim(meta, word_embed, tiled_intent_embed)

                    all_intents_embed_in = tf.placeholder(tf.float32, (None, None, meta['embed_dim']),
                                                          name='all_intents_embed')
                    sim_all = cls._tf_gpu_sim(meta, word_embed, all_intents_embed_in)

                    saver = tf.train.Saver()

                else:
                    saver = tf.train.import_meta_graph(checkpoint + '.meta')

                    # iterator = tf.get_collection('data_iterator')[0]

                    a_in = tf.get_collection('message_placeholder')[0]
                    b_in = tf.get_collection('intent_placeholder')[0]

                    sim_op = tf.get_collection('similarity_op')[0]

                    all_intents_embed_in = tf.get_collection('all_intents_embed_in')[0]
                    sim_all = tf.get_collection('sim_all')[0]

                    word_embed = tf.get_collection('word_embed')[0]
                    intent_embed = tf.get_collection('intent_embed')[0]

                saver.restore(sess, checkpoint)

            with io.open(os.path.join(
                    model_dir,
                    file_name + "_inv_responses_dict.pkl"), 'rb') as f:
                inv_intent_dict = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    file_name + "_encoded_all_responses.pkl"), 'rb') as f:
                encoded_all_intents = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    file_name + "_all_responses_embed_values.pkl"), 'rb') as f:
                all_intents_embed_values = pickle.load(f)

            return cls(
                component_config=meta,
                inv_response_dict=inv_intent_dict,
                encoded_all_responses=encoded_all_intents,
                all_responses_embed_values=all_intents_embed_values,
                session=sess,
                graph=graph,
                message_placeholder=a_in,
                response_placeholder=b_in,
                similarity_op=sim_op,
                all_responses_embed_in=all_intents_embed_in,
                sim_all=sim_all,
                word_embed=word_embed,
                response_embed=intent_embed
            )

        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return cls(component_config=meta)
