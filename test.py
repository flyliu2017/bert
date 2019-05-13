import os
import numpy as np


import modeling
import tokenization
import tensorflow as tf
from tensorflow.contrib.tpu import TPUEstimator

from data_processor import PaddingInputExample
from model_fn import model_fn_builder
from utils import report_metrics,precision_recall_f1score
from sequential_tag_processor import *

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "data_converted", True,
    "Whether data had been converted to tfrecord.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")
flags.DEFINE_integer(
    "num_gpus", 1,
    "number of GPU to use.")
flags.DEFINE_float(
    "threthold", None,
    "probability threshold.")
flags.DEFINE_integer(
    "threthold_num", 20,
    "threshold num.")

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "phrase": ExtractPhrasesProcessor,
        "seg-phrase": ExtractPhrasesFromSegmentedInputProcessor,
        "all-phrase": ExtractAllPhrasesProcessor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    processor = processors[task_name](FLAGS.data_dir,tokenizer,FLAGS.max_seq_length)

    label_list = processor.get_labels()


    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples()
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    threthold = None

    model_fn = model_fn_builder(
        processor,
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    # strategy=tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.num_gpus)
    # run_config = tf.estimator.RunConfig(
    #         model_dir=FLAGS.output_dir,
    #         save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    #         train_distribute=strategy
    #         )
    # estimator = tf.estimator.Estimator(
    #     model_fn=model_fn,
    #     config=run_config,
    #     params={'batch_size':FLAGS.train_batch_size}
    #     )

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if not tf.gfile.Exists(train_file) or not FLAGS.data_converted:
            processor.file_based_convert_examples_to_features(
                train_examples, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = processor.file_based_input_fn_builder(
            input_file=train_file,
            is_training=True,
            drop_remainder=True)
        train_hook = tf.train.LoggingTensorHook(['loss/train_loss', 'lr'], every_n_iter=100)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[train_hook])

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples()
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        if not tf.gfile.Exists(eval_file) or not FLAGS.data_converted:
            processor.file_based_convert_examples_to_features(
                eval_examples,eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = processor.file_based_input_fn_builder(
            input_file=eval_file,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        # result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        result = estimator.predict(input_fn=eval_input_fn)

        label_ids = []
        probabilities = []
        for (i, prediction) in enumerate(result):
            # if i >= num_actual_predict_examples:
            #     break
            probabilities.append(prediction["probabilities"])
            label_ids.append(list(prediction['label_ids']))

        threthold_num = FLAGS.threthold_num
        metrics = [precision_recall_f1score(label_id, prob, threthold_num=threthold_num) for label_id, prob in
                   zip(label_ids, probabilities)]
        precisions, recalls, f1scores = list(zip(*metrics))

        mean_f1scores = np.mean(f1scores, axis=0)
        index = np.argmax(mean_f1scores)
        best_f1score = mean_f1scores[index]
        best_precision = np.mean(precisions, axis=0)[index]
        best_recall = np.mean(recalls, axis=0)[index]

        threthold = index / threthold_num

        tags = np.array(probabilities) >= threthold
        label_ids = np.array(label_ids) == 1

        equal = [n.all() for n in tags == label_ids]
        accuracy = sum(equal) / len(equal)

        metrics_result = {}
        metrics_result['accuracy'] = accuracy
        metrics_result['best_f1score'] = best_f1score
        metrics_result['best_precision'] = best_precision
        metrics_result['best_recall'] = best_recall
        metrics_result['threthold'] = threthold

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            writer.write("threthold: {}\n".format(threthold))
            for key in sorted(metrics_result.keys()):
                tf.logging.info("  %s = %s", key, str(metrics_result[key]))
                writer.write("%s = %s\n" % (key, str(metrics_result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples()
        # num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        if not tf.gfile.Exists(predict_file) or not FLAGS.data_converted:
            processor.file_based_convert_examples_to_features(predict_examples,predict_file)

        tf.logging.info("***** Running prediction*****")
        # tf.logging.info("  Num examples = %d (%d actual, %d padding)",
        #                 len(predict_examples), num_actual_predict_examples,
        #                 len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = processor.file_based_input_fn_builder(
            input_file=predict_file,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")

        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tag_list = []
            label_ids = []
            if FLAGS.threthold:
                threthold = FLAGS.threthold
            elif threthold is None:
                threthold = 0.5

            tf.logging.info("***** Predict results *****")

            for (i, prediction) in enumerate(result):
                # if i >= num_actual_predict_examples:
                #     break
                probabilities = prediction["probabilities"]
                tag = [1 if n >= threthold else 0 for n in probabilities]
                # tag=[0]+tag[:-1]

                tag_list.append(tag)
                label_ids.append(list(prediction['label_ids']))

                input_ids = prediction['input_ids']
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                phrase = [tokens[j] if tag[j] == 1 else ' ' for j in range(min(len(tokens), 128))]

                phrase = ''.join(phrase).strip()
                writer.write(phrase + '\n')
                num_written_lines += 1

            report = report_metrics(tag_list, label_ids)
            writer.write("threthold: {}\n".format(threthold))
            writer.write(report)

        # assert num_written_lines == num_actual_predict_examples
        tf.logging.info("threthold: {}".format(threthold))
        tf.logging.info(report)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()