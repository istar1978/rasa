import sys
import os
import tensorflow as tf
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def sparsity_report(graph, sess):
    mask_names = [
        v.name
        for v in graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        if v.name.endswith("/mask:0")
    ]
    mask_names = sorted(mask_names)

    nnz = 0.0
    nnz_column = 0.0
    total = 0.0
    total_columns = 0.0

    for mask_name in mask_names:
        mask_tensor = graph.get_tensor_by_name(mask_name)
        mask = sess.run(mask_tensor)
        nnz_local = np.count_nonzero(mask)
        nnz_column_local = np.count_nonzero(np.sum(mask, axis=0))
        total_local = mask.size
        total_columns_local = mask.shape[1]

        nnz += nnz_local
        nnz_column += nnz_column_local
        total += total_local
        total_columns += total_columns_local

        sparsity_local = 0 if nnz_local == 0 else 100 * (1 - nnz_local / total_local)
        sparsity_column_local = 100 * (1 - nnz_column_local / total_columns_local)
        print (
            "Variable: {}\n\tShape: {}\n\tElement sparsity: {:.1f}%\n\tColumn sparsity: {:.1f}% ({}/{})".format(
                mask_name,
                mask.shape,
                sparsity_local,
                sparsity_column_local,
                nnz_column_local,
                total_columns_local,
            )
        )

    element_sparsity = 0 if nnz == 0 else 100 * (1 - nnz / total)
    column_sparsity = 0 if nnz_column == 0 else 100 * (1 - nnz_column / total_columns)
    print (
        "\n###########################################################################"
    )
    print (
        "Overall:\n\tElement sparsity: {:.1f}%\n\tColumn sparsity: {:.1f}%".format(
            element_sparsity, column_sparsity
        )
    )


def resize_bert_weights(graph, sess, tmp_ckpt_name="tmp/bert-np-resized/model.ckpt"):
    vars = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    new_vars = []
    masks = [v for v in vars if v.name.endswith("/mask:0")]

    # resize each weight matrix and bias vector that correspond to a pruning mask
    for mask in masks:
        # print("Resizing from mask: {}".format(mask.name))
        scope = "/".join(mask.name.split("/")[:-1])

        mask_squashed = np.amax(sess.run(mask), axis=0)
        nonzero = np.count_nonzero(mask_squashed)
        sparsity = 1 - (nonzero / len(mask_squashed))
        # print(
        #     "sparsity: {:.2f} ({}/{} nonzero)".format(
        #         sparsity, nonzero, len(mask_squashed)
        #     )
        # )

        weights_name = scope + "/weights:0"
        biases_name = scope + "/biases:0"
        weights_name_new = scope + "/kernel"
        biases_name_new = scope + "/bias"

        weights = graph.get_tensor_by_name(weights_name)
        biases = graph.get_tensor_by_name(biases_name)

        # do "cross-pruning" only for the encoder layer output weight matrices,
        # based on intermediate layer activation sparsity
        if mask.name.startswith("bert/encoder/") and mask.name == (
            "/".join(scope.split("/")[:3]) + "/output/dense/mask:0"
        ):
            incoming_tensor_mask_name = (
                "/".join(scope.split("/")[:3]) + "/intermediate/dense/mask:0"
            )
            incoming_tensor_mask = graph.get_tensor_by_name(incoming_tensor_mask_name)
            incoming_tensor_mask_squashed = np.amax(
                sess.run(incoming_tensor_mask), axis=0
            )
            incoming_tensor_sparsity = 1 - (
                np.count_nonzero(incoming_tensor_mask_squashed)
                / len(incoming_tensor_mask_squashed)
            )

            weights = sess.run(weights)[incoming_tensor_mask_squashed > 0, :]
            weights = weights[:, mask_squashed > 0]
        else:
            weights = sess.run(weights)[:, mask_squashed > 0]

        biases = sess.run(biases)[mask_squashed > 0]

        new_vars.append((biases, biases_name_new, True))
        new_vars.append((weights, weights_name_new, True))
        new_vars.append((sess.run(mask), mask.name, False))

    # add other variables that need to be saved
    batchnorm_vars = [
        var
        for var in vars
        if (var.name.endswith("/beta:0") or var.name.endswith("/gamma:0"))
    ]
    embedding_vars = [
        var
        for var in vars
        if var.name
        in [
            "bert/embeddings/word_embeddings:0",
            "bert/embeddings/token_type_embeddings:0",
            "bert/embeddings/position_embeddings:0",
        ]
    ]
    output_vars = [
        var for var in vars if var.name in ["output_weights:0", "output_bias:0"]
    ]
    vars_to_keep = batchnorm_vars + embedding_vars + output_vars
    for var in vars_to_keep:
        value = sess.run(var.name)
        new_vars.append((value, var.name, True))

    # save the resized weight matrices and bias vectors, plus other necessary variables
    new_graph = tf.Graph()
    with new_graph.as_default():
        for (value, name, is_trainable) in new_vars:
            new_var = tf.Variable(
                initial_value=value, trainable=is_trainable, name=name.split(":")[0]
            )

        init = tf.initializers.global_variables()
        saver = tf.train.Saver()
        new_sess = tf.Session(graph=new_graph)
        new_sess.run(init)
        save_path = saver.save(
            new_sess, tmp_ckpt_name, global_step=0, write_meta_graph=False
        )

    return save_path


def fake_quantise_tf_variables(vars, n_clusters, graph, sess):
    with graph.as_default():
        quantisation_init_ops = []

        for name in vars:
            print ("Quantising '{}'...".format(name))
            scope = "/".join(name.split("/")[:-1])
            var = [
                v
                for v in graph.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope
                )
                if v.name == name
            ][0]

            original_shape = var.shape
            weights_numpy = sess.run(var)

            unique_vals = np.unique(weights_numpy.flatten())
            if len(unique_vals) < n_clusters:
                print (
                    "Number of unique values ({}) is less than the number of clusters ({}), taking n_clusters={}.".format(
                        len(unique_vals), n_clusters, len(unique_vals)
                    )
                )
                n_clusters = len(unique_vals)

            weights_indices, cluster_centres = fake_quantise_np_array(
                weights_numpy, n_clusters, var.name.split(":")[0]
            )
            weights_quantised_values = tf.gather(
                params=cluster_centres,
                indices=weights_indices,
                name=name.split(":")[0] + "_indices_to_cluster_centres",
            )
            sess.run(tf.initialize_variables([cluster_centres, weights_indices]))

            weights_quantised_values = tf.reshape(
                weights_quantised_values, original_shape
            )
            quantise_init_op = tf.assign(
                var,
                weights_quantised_values,
                name=name.split(":")[0] + "_replace_by_quantised",
            )
            quantisation_init_ops.append(quantise_init_op)

        sess.run(quantisation_init_ops)


def fake_quantise_np_array(arr, n_clusters, name):
    arr_flat = arr.flatten().reshape((-1, 1))

    # initialise cluster centres
    min_weight, max_weight = (
        np.min(arr_flat).item() * 10,
        np.max(arr_flat).item() * 10,
    )
    spacing = (max_weight - min_weight) / (n_clusters + 1)
    init_centre_values = (
        np.linspace(min_weight + spacing, max_weight - spacing, n_clusters) / 10
    )
    init_centre_values = init_centre_values.reshape(-1, 1)

    # cluster values
    np.random.seed(42)
    kmeans = MiniBatchKMeans(
        n_clusters,
        init=init_centre_values,
        n_init=1,
        max_iter=100,
        init_size=max(300, 3 * n_clusters),
        verbose=False,
    )
    kmeans.fit(arr_flat)

    # turn weights into pointers to cluster centres
    arr_as_centre_indices = kmeans.predict(arr_flat)
    centres = np.array([centre[0] for centre in kmeans.cluster_centers_])

    # create TF graph elements
    centres_tf_var = tf.get_variable(
        name + "-centres", dtype=tf.float32, initializer=tf.constant(centres)
    )
    weights_as_centre_indices = tf.get_variable(
        name + "-pointers_to_centres",
        dtype=tf.int32,
        initializer=tf.constant(arr_as_centre_indices.astype(np.int32)),
        trainable=False,
    )

    return weights_as_centre_indices, centres_tf_var
