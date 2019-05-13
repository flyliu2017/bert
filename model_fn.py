from tensorflow.python.ops.losses.losses_impl import Reduction

import modeling
import tensorflow as tf

import optimization


def create_sequential_tagging_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias",[], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.reduce_sum(tf.multiply(output_layer,output_weights),-1)
        logits = tf.add(logits, output_bias)

        probabilities=tf.sigmoid(logits)
        input_mask = tf.cast(input_mask, tf.float32)
        probabilities=tf.multiply(probabilities,input_mask)
        # labels=tf.constant(labels,dtype=tf.int32)

        per_example_loss=tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits,reduction=Reduction.NONE)
        per_example_loss=tf.multiply(per_example_loss,input_mask)
        per_example_loss=tf.reduce_sum(per_example_loss,axis=-1)
        loss = tf.reduce_mean(per_example_loss,name='train_loss')

        return (loss, per_example_loss, logits, probabilities)

def model_fn_builder(processor,bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(label_ids.shape[0], dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = processor.create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        tf.summary.scalar('learning_rate',learning_rate)

        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                # ones=tf.get_variable('ones',shape=logits.shape,initializer=tf.ones_initializer)
                # zeros=tf.get_variable('zeros',shape=logits.shape,initializer=tf.zeros_initializer)
                predictions=tf.where(logits>=0,tf.ones(tf.shape(logits)),tf.zeros(tf.shape(logits)))
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities,
                             'input_ids':input_ids,
                             'label_ids':label_ids},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn