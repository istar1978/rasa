# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
from rasa.nlu.classifiers.bert.optimization import create_optimizer
from rasa.nlu.classifiers.bert.tokenization import convert_to_unicode, FullTokenizer
from rasa.nlu.classifiers.bert.modeling import BertModel

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib.model_pruning.python import pruning as magnitude_pruning
from tensorflow.contrib.model_pruning.python.layers import core_layers as core
from rasa.nlu.classifiers.fake_neuron_pruning import pruning as neuron_pruning


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self, input_ids, input_mask, segment_ids, label_id, is_real_example=True
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False,
        )

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0 : (max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    """
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    """

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True,
    )
    return feature


#
#
#


def get_train_examples(training_examples):
    """See base class."""
    return create_examples(training_examples, "train")


def get_test_examples(training_examples):
    """See base class."""
    return create_examples(training_examples, "test")


def get_labels(training_data):
    """See base class."""
    return sorted(
        set([example.get("intent") for example in training_data.intent_examples])
    )


def create_examples(rasa_training_examples, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, rasa_example) in enumerate(rasa_training_examples):
        guid = "%s-%s" % (set_type, i)
        line = convert_to_unicode(rasa_example.text)
        line = line.strip()
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        label = convert_to_unicode(rasa_example.data["intent"])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples


#
#
#


def check_global_sparsity():
    """Add a summary for the weight sparsity."""
    weight_masks = magnitude_pruning.get_masks()
    weights_per_layer = []
    nonzero_per_layer = []
    for mask in weight_masks:
        nonzero_per_layer.append(tf.reduce_sum(mask))
        weights_per_layer.append(tf.size(mask))
        total_nonzero = tf.add_n(nonzero_per_layer)
        total_weights = tf.add_n(weights_per_layer)
    sparsity = 1.0 - (
        tf.cast(total_nonzero, tf.float32) / tf.cast(total_weights, tf.float32)
    )
    tf.summary.scalar("global_weight_sparsity", sparsity)


def build_sparsity_map(
    graph,
    keys=0.5,
    queries=0.5,
    values=0.5,
    att_outputs=0.5,
    intermediates=0.5,
    outputs=0.5,
    pooler=0.5,
):
    weights = graph.get_collection(core.WEIGHT_COLLECTION)
    sparsity_map = {}
    for w in weights:
        # print(w.name)
        if "/key/weights" in w.name:
            sparsity = keys
        elif "/query/weights" in w.name:
            sparsity = queries
        elif "/value/weights" in w.name:
            sparsity = values
        elif "/attention/output/dense/weights" in w.name:
            sparsity = att_outputs
        elif "/intermediate/dense/weights" in w.name:
            sparsity = intermediates
        elif "/output/dense/weights" in w.name and "attention" not in w.name:
            sparsity = outputs
        elif "/pooler/dense/weights" in w.name:
            sparsity = pooler
        else:
            raise ValueError("Unrecognised weight to be pruned: {}".format(w.name))
        sparsity_map[w.name] = sparsity

    return sparsity_map


def load_pruning_masks_from_ckpt(ckpt_name):
    ckpt_reader = tf.train.NewCheckpointReader(ckpt_name)
    all_names = ckpt_reader.get_variable_to_shape_map().keys()
    mask_names = [n for n in all_names if n.endswith("/mask")]
    mask_names.sort()
    mask_dict = {}

    for mask_name in mask_names:
        mask = ckpt_reader.get_tensor(mask_name)
        scope = "/".join(mask_name.split("/")[:-1])
        mask_squashed = np.amax(mask, axis=0)
        nonzero = np.count_nonzero(mask_squashed)
        sparsity = 1 - (nonzero / len(mask_squashed))
        mask_dict[scope] = mask_squashed
        # print ("{}: {:.1f}% sparsity".format(scope, 100 * sparsity))

    return mask_dict


def pruning_hparams(hparams):  # pylint: disable=unused-argument
    """Helper to get hparams for pruning library."""
    hparams = tf.contrib.training.HParams(
        begin_pruning_step=hparams.get("begin_pruning_step"),
        end_pruning_step=hparams.get("end_pruning_step"),
        pruning_frequency=hparams.get("pruning_frequency"),
        target_sparsity=hparams.get("target_sparsity"),
        sparsity_function_begin_step=hparams.get("begin_pruning_step"),
        sparsity_function_end_step=hparams.get("end_pruning_step"),
        # nbins=hparams.get("nbins"),
        weight_sparsity_map=[""],
        name="model_pruning",
        threshold_decay=0.0,
        initial_sparsity=0.0,
        sparsity_function_exponent=1,
        use_tpu=False,
        block_height=1,
        block_width=1,
        block_pooling_function="AVG",
    )

    return hparams

    # def model_fn_builder(
    #     bert_tfhub_module_handle,
    #     num_labels,
    #     learning_rate,
    #     num_train_steps,
    #     num_warmup_steps,
    #     bert_config,
    # ):
    """Returns `model_fn` closure for TPUEstimator."""


def build_model(
    features,
    mode,
    params,
    bert_tfhub_module_handle,
    num_labels,
    learning_rate,
    num_train_steps,
    num_warmup_steps,
    bert_config,
):  # pylint: disable=unused-argument
    print ("####################\nMODE=<{}>\n##########################".format(mode))
    # print(tf.estimator.ModeKeys.PREDICT)
    """The `model_fn` for TPUEstimator."""
    input_ids = tf.identity(features["input_ids"], name="input_ids")
    input_mask = tf.identity(features["input_mask"], name="input_mask")
    segment_ids = tf.identity(features["segment_ids"], name="segment_ids")
    label_ids = tf.identity(features["label_ids"], name="label_ids")

    # print(input_ids, input_mask, segment_ids, label_ids)
    # exit(0)

    input_tensors = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "label_ids": label_ids,
    }

    is_predicting = mode == "predict"
    if not is_predicting:
        masks_dict = None
        if (
            params["sparsity_technique"] == "neuron_pruning"
            and params["sparsification_params"]["resize_pruned_matrices"]
        ):
            # get a var: mask dict from checkpoint
            masks_ckpt = params["sparsification_params"]["checkpoint_for_pruning_masks"]
            if masks_ckpt is None:
                raise ValueError(
                    "You are trying to resize pruned weight matrices but haven't provided a checkpoint to take the masks from!"
                )
            masks_dict = load_pruning_masks_from_ckpt(ckpt_name=masks_ckpt)

        g = tf.get_default_graph()
        (loss, predicted_labels, log_probs) = create_model(
            is_predicting,
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            num_labels,
            bert_tfhub_module_handle,
            bert_config,
            sparsity_technique=params["sparsity_technique"],
            trained_masks=masks_dict,
        )
        with g.as_default():
            with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predicted_labels, name="accuracy"
                )
        # [print(v.name) for v in g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        # exit(0)
        """
        logging_hook = tf.train.LoggingTensorHook(
            {"loss": loss, "accuracy": accuracy[1]}, every_n_iter=1
        )

        # Calculate evaluation metrics.
        def metric_fn(label_ids, predicted_labels):
            accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
            f1_score = tf.contrib.metrics.f1_score(label_ids, predicted_labels)
            auc = tf.metrics.auc(label_ids, predicted_labels)
            recall = tf.metrics.recall(label_ids, predicted_labels)
            precision = tf.metrics.precision(label_ids, predicted_labels)
            true_pos = tf.metrics.true_positives(label_ids, predicted_labels)
            true_neg = tf.metrics.true_negatives(label_ids, predicted_labels)
            false_pos = tf.metrics.false_positives(label_ids, predicted_labels)
            false_neg = tf.metrics.false_negatives(label_ids, predicted_labels)
            return {
                "eval_accuracy": accuracy,
                "f1_score": f1_score,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "true_positives": true_pos,
                "true_negatives": true_neg,
                "false_positives": false_pos,
                "false_negatives": false_neg,
            }

        eval_metrics = metric_fn(label_ids, predicted_labels)
        """
        if mode == "train":
            if params["sparsity_technique"] == "weight_pruning":
                train_op = create_optimizer(
                    loss,
                    learning_rate,
                    num_train_steps,
                    num_warmup_steps,
                    use_tpu=False,
                    vars_to_optimize=(
                        None
                        if not params["finetune_hat_only"]
                        else ["output_weights", "output_bias"]
                    ),
                )
                if not "load_masks_from" in params:
                    # If we are loading trained masks, don't add the mask update
                    # step to the training process and keep the masks static
                    with tf.control_dependencies([train_op]):
                        mp_hparams = pruning_hparams(params["sparsification_params"])
                        p = magnitude_pruning.Pruning(
                            mp_hparams, global_step=tf.train.get_global_step()
                        )
                        mask_update_op = p.conditional_mask_update_op()
                        train_op = mask_update_op
                check_global_sparsity()

            elif params["sparsity_technique"] == "neuron_pruning":
                # prepare gradient and activation accumulators
                # accumulate activations and grads by summing, this basically requires two ops to be added to train_op
                # step through all relevant vars
                # multiply activations by gradients, sum as required to get neuron scores
                # select K smallest neurons
                # update the weight matrix by setting stuff to 0

                """
                def train_formatter(stats):
                    s = "accuracy = {:.4f}, loss = {:.5f}, sparsity = {:.3f}, train_step = {}".format(
                        stats["accuracy"],
                        stats["loss"],
                        stats["sparsity"],
                        stats["train_step"],
                    )
                    return s
                """

                if not params["sparsification_params"][
                    "resize_pruned_matrices"
                ]:  # doing pruning mask updates and setting up all tha fancy shit around that
                    pruning_activations = g.get_collection(
                        "pruning_activation_providers"
                    )
                    with tf.name_scope("neuron_pruning_training"):
                        grads = tf.gradients(
                            loss, pruning_activations, name="gradients_for_pruning"
                        )

                        for gradient, activation in zip(grads, pruning_activations):
                            scope = "/".join(activation.name.split("/")[:-1])
                            neuron_rank_accumulator_name = (
                                scope + "/neuron_rank_accumulator:0"
                            )
                            neuron_rank_accumulator = [
                                v
                                for v in g.get_collection("neuron_rank_accumulators")
                                if v.name == neuron_rank_accumulator_name
                            ][0]

                            with tf.name_scope(scope + "/rank"):
                                local_rank = tf.multiply(gradient, activation)
                                local_rank = tf.math.reduce_mean(local_rank, axis=0)
                                neuron_rank_accumulator_update_op = tf.assign_add(
                                    neuron_rank_accumulator,
                                    local_rank,
                                    name="neuron_rank_accumulator_update",
                                )
                                g.add_to_collection(
                                    "neuron_rank_accumulators_update",
                                    neuron_rank_accumulator_update_op,
                                )

                        update_accumulators_op = tf.group(
                            g.get_collection("neuron_rank_accumulators_update"),
                            name="all_accumulators_update_op",
                        )

                        train_op = create_optimizer(
                            loss,
                            learning_rate,
                            num_train_steps,
                            num_warmup_steps,
                            use_tpu=False,
                            vars_to_optimize=(
                                None
                                if not params["finetune_hat_only"]
                                else ["output_weights", "output_bias"]
                            ),
                        )

                        # sparsity_map = build_sparsity_map(g)
                        # sparsity_map = build_sparsity_map(
                        #     g,
                        #     keys=0.9,
                        #     queries=0.9,
                        #     values=0.45,
                        #     att_outputs=0.25,
                        #     intermediates=0.45,
                        #     outputs=0.2,
                        #     pooler=0.3,
                        # )
                        sparsity_map = build_sparsity_map(
                            g,
                            keys=0.9,
                            queries=0.9,
                            values=0.45,
                            att_outputs=0.25,
                            intermediates=0.6,
                            outputs=0.2,
                            pooler=0.3,
                        )
                        # c = 1.2
                        # sparsity_map = build_sparsity_map(
                        #     g,
                        #     keys=0.9,
                        #     queries=0.9,
                        #     values=0.45 * c,
                        #     att_outputs=0.25 * c,
                        #     intermediates=0.6,
                        #     outputs=0.2 * c,
                        #     pooler=0.3 * c,
                        # )

                        # sparsity_map = build_sparsity_map(
                        #     g,
                        #     keys=almost_zero,
                        #     queries=almost_zero,
                        #     values=almost_zero,
                        #     att_outputs=almost_zero,
                        #     intermediates=almost_zero,
                        #     outputs=almost_zero,
                        #     pooler=almost_zero,
                        # )
                        # print (sparsity_map)
                        # exit(0)

                        with tf.control_dependencies([train_op]):
                            with tf.control_dependencies([update_accumulators_op]):
                                mp_hparams = pruning_hparams(
                                    params["sparsification_params"]
                                )
                                pruning = neuron_pruning.Pruning(
                                    mp_hparams,
                                    global_step=tf.train.get_global_step(),
                                    neuron_rank_accumulators_reset="accumulators_reset",
                                    sparsity_map=sparsity_map,
                                )
                                mask_update_op = pruning.conditional_mask_update_op()
                                train_op = mask_update_op
                    """
                    logging_hook = tf.train.LoggingTensorHook(
                        {
                            "loss": loss,
                            "accuracy": accuracy[1],
                            "train_step": tf.train.get_global_step(),
                            "sparsity": pruning._get_sparsity(),
                        },
                        every_n_iter=1,
                        formatter=train_formatter,
                    )
                    """
                else:  # not doing pruning mask updates
                    print ("Resizing the weight matrices!")
                    train_op = create_optimizer(
                        loss,
                        learning_rate,
                        num_train_steps,
                        num_warmup_steps,
                        use_tpu=False,
                        vars_to_optimize=(
                            None
                            if not params["finetune_hat_only"]
                            else ["output_weights", "output_bias"]
                        ),
                    )
                    """
                    logging_hook = tf.train.LoggingTensorHook(
                        {
                            "loss": loss,
                            "accuracy": accuracy[1],
                            "train_step": tf.train.get_global_step(),
                            "sparsity": params["sparsification_params"]["target_sparsity"]
                        },
                        every_n_iter=1,
                        formatter=train_formatter,
                    )
                    """
                """
                writer = tf.summary.FileWriter(
                    logdir="tfgraph-bert-raw-train-np", graph=g
                )
                writer.flush()
                # exit(0)
                """

            else:
                train_op = create_optimizer(
                    loss,
                    learning_rate,
                    num_train_steps,
                    num_warmup_steps,
                    use_tpu=False,
                    vars_to_optimize=(
                        None
                        if not params["finetune_hat_only"]
                        else ["output_weights", "output_bias"]
                    ),
                )

            return train_op, loss, input_tensors, log_probs, predicted_labels, accuracy
        else:
            raise ValueError("'mode' must be either 'train' or 'predict'")
    else:
        # print ("DOING INFERENCE MODEL PREPARATION")
        # print (params)
        # print (input_ids.shape)
        if params["sparsity_technique"] != "neuron_pruning":
            (predicted_labels, log_probs) = create_model(
                is_predicting,
                input_ids,
                input_mask,
                segment_ids,
                label_ids,
                num_labels,
                bert_tfhub_module_handle,
                bert_config,
                sparsity_technique=params["sparsity_technique"],
            )
        elif params["sparsity_technique"] == "neuron_pruning":
            masks_dict = None
            if params["sparsification_params"]["resize_pruned_matrices"]:
                masks_ckpt = params["sparsification_params"][
                    "checkpoint_for_pruning_masks"
                ]
                if masks_ckpt is None:
                    raise ValueError(
                        "You are trying to resize pruned weight matrices but haven't provided a checkpoint to take the masks from!"
                    )
                masks_dict = load_pruning_masks_from_ckpt(ckpt_name=masks_ckpt)

            (predicted_labels, log_probs) = create_model(
                is_predicting,
                input_ids,
                input_mask,
                segment_ids,
                label_ids,
                num_labels,
                bert_tfhub_module_handle,
                bert_config,
                sparsity_technique=params["sparsity_technique"],
                trained_masks=masks_dict,
            )
        else:
            raise ValueError("WTF??")

        return predicted_labels, log_probs, input_tensors


def create_model(
    is_predicting,
    input_ids,
    input_mask,
    segment_ids,
    labels,
    num_labels,
    bert_tfhub_module_handle=None,
    bert_config=None,
    use_one_hot_embeddings=True,
    sparsity_technique="weight_pruning",
    trained_masks=None,
):
    """Creates a classification model."""
    if bert_config:
        model = BertModel(
            config=bert_config,
            is_training=not is_predicting,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
            sparsity_technique=sparsity_technique,
            trained_np_masks=trained_masks,
            scope="bert",
        )

        output_layer = model.get_pooled_output()
    else:
        if sparsity_technique is not None:
            raise ValueError(
                "Trying to use sparsity technique '{}' with a model from TF Hub is not supported.".format(
                    sparsity_technique
                )
            )

        bert_module = hub.Module(bert_tfhub_module_handle, trainable=True)
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

        output_layer = bert_outputs["pooled_output"]

    hidden_size = output_layer.shape[-1].value

    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        output_weights = tf.get_variable(
            "output_weights",
            [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer()
        )

    with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
        if not is_predicting:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1, name="log_probs")

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(
            tf.argmax(log_probs, axis=-1, output_type=tf.int32), name="predictions"
        )

        if is_predicting:
            return (predicted_labels, log_probs)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)


def create_tokenizer_from_hub_module(bert_tfhub_module_handle):
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(bert_tfhub_module_handle)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run(
                [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
            )
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


#
#
#


def serving_input_fn_builder(max_seq_length, is_predicting=False):
    if is_predicting:
        print ("CREATING PREDICTING INPUT SERVER (BATCH SIZE = 1)")

        def serving_input_fn():
            label_ids = tf.placeholder(tf.int32, [None], name="label_ids")
            input_ids = tf.placeholder(
                tf.int32, [None, max_seq_length], name="input_ids"
            )
            input_mask = tf.placeholder(
                tf.int32, [None, max_seq_length], name="input_mask"
            )
            segment_ids = tf.placeholder(
                tf.int32, [None, max_seq_length], name="segment_ids"
            )
            input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
                {
                    "label_ids": label_ids,
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": segment_ids,
                },
                default_batch_size=1,
            )()
            return input_fn

    else:

        def serving_input_fn():
            label_ids = tf.placeholder(tf.int32, [None], name="label_ids")
            input_ids = tf.placeholder(
                tf.int32, [None, max_seq_length], name="input_ids"
            )
            input_mask = tf.placeholder(
                tf.int32, [None, max_seq_length], name="input_mask"
            )
            segment_ids = tf.placeholder(
                tf.int32, [None, max_seq_length], name="segment_ids"
            )
            input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
                {
                    "label_ids": label_ids,
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": segment_ids,
                }
            )()
            return input_fn

    return serving_input_fn


#
#
#


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def build_input_dataset(features, seq_length, is_training, drop_remainder, params):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    # def input_fn(params):
    #     """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices(
        {
            "input_ids": tf.constant(
                all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32
            ),
            "input_mask": tf.constant(
                all_input_mask, shape=[num_examples, seq_length], dtype=tf.int32
            ),
            "segment_ids": tf.constant(
                all_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32
            ),
            "label_ids": tf.constant(
                all_label_ids, shape=[num_examples], dtype=tf.int32
            ),
        }
    )

    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

    # return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices(
            {
                "input_ids": tf.constant(
                    all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32
                ),
                "input_mask": tf.constant(
                    all_input_mask, shape=[num_examples, seq_length], dtype=tf.int32
                ),
                "segment_ids": tf.constant(
                    all_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32
                ),
                "label_ids": tf.constant(
                    all_label_ids, shape=[num_examples], dtype=tf.int32
                ),
            }
        )

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""
    features = []
    for (ex_index, example) in enumerate(examples):
        feature = convert_single_example(
            ex_index, example, label_list, max_seq_length, tokenizer
        )
        features.append(feature)
    return features
