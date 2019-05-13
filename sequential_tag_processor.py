import os
import csv
import tensorflow as tf

import tokenization
from data_processor import DataProcessor, InputExample, InputFeatures
from model_fn import create_sequential_tagging_model


class SequentialTagProcessor(DataProcessor):
    """Base class for data converters for sequence classification data sets."""

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    @property
    def name_to_features(self): 
        return  {
              "input_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
              "input_mask": tf.FixedLenFeature([self.max_seq_length], tf.int64),
              "segment_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
              "label_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
              "is_real_example": tf.FixedLenFeature([], tf.int64),
        }

    @property
    def padding_input_features(self):
        return InputFeatures(
            input_ids=[0] * self.max_seq_length,
            input_mask=[0] * self.max_seq_length,
            segment_ids=[0] * self.max_seq_length,
            label_id=[0] * self.max_seq_length,
            is_real_example=False)

    @property
    def create_model(self):
        return create_sequential_tagging_model

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def create_label_features(self, example, tokens):
        label = example.label
        if not isinstance(label, list):
            label = [label]
        label_tokens = [self.tokenizer.tokenize(l) for l in label]
        label_id = [0] * self.max_seq_length
        for label in label_tokens:
            length = len(label)
            for i in range(len(tokens) - length + 1):
                if tokens[i:i + length] == label:
                    start = i
                    end = i + length
                    label_id[start:end] = [1] * (end - start)
                    break

            else:
                raise ValueError("can't find phrase in text.")
        return label_id

class ExtractPhrasesProcessor(SequentialTagProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self):
        """See base class."""
        return self._create_examples( "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples( "eval")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples("test")



    def _create_examples(self, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(os.path.join(self.data_dir, '{}_xs_converted_tag.txt'.format(set_type)), 'r', encoding='utf8') as f:
            txts = f.read().splitlines()
        with open(os.path.join(self.data_dir, '{}_ys_converted_tag.txt'.format(set_type)), 'r', encoding='utf8') as f:
            labels = f.read().splitlines()

        for (i, n) in enumerate(zip(txts, labels)):
            txt, label = n
            guid = "%s-%s" % (set_type, i)
            text_a,text_b=txt.split(' | ')
            label = label.split(' | ')[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class ExtractPhrasesFromSegmentedInputProcessor(ExtractPhrasesProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def _create_examples(self,set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(os.path.join(self.data_dir, '{}_xs_converted_tag.txt'.format(set_type)), 'r', encoding='utf8') as f:
            txts = f.read().splitlines()
        with open(os.path.join(self.data_dir, '{}_ys_converted_tag.txt'.format(set_type)), 'r', encoding='utf8') as f:
            labels = f.read().splitlines()

        for (i, n) in enumerate(zip(txts, labels)):
            txt, label = n
            guid = "%s-%s" % (set_type, i)
            text_a,text_b=txt.split(' | ')
            text_a=' '.join(list(text_a))
            label = label.split(' | ')[0]
            label=' '.join(list(label))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

class ExtractPhrasesTagPrefixedProcessor(ExtractPhrasesFromSegmentedInputProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def _create_examples(self,  set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(os.path.join(self.data_dir, '{}_xs_prefix.txt'.format(set_type)), 'r', encoding='utf8') as f:
            txts = f.read().splitlines()
        with open(os.path.join(self.data_dir, '{}_ys_converted_tag.txt'.format(set_type)), 'r', encoding='utf8') as f:
            labels = f.read().splitlines()

        for (i, n) in enumerate(zip(txts, labels)):
            txt, label = n
            guid = "%s-%s" % (set_type, i)
            text_a,text_b=txt.split(' | ')
            text_b=' '.join(list(text_b))
            label = label.split(' | ')[0]
            label=' '.join(list(label))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

class ExtractAllPhrasesProcessor(ExtractPhrasesFromSegmentedInputProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def _create_examples(self,  set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(os.path.join(self.data_dir, '{}_xs_multitags'.format(set_type)), 'r', encoding='utf8') as f:
            txts = f.read().splitlines()
        with open(os.path.join(self.data_dir, '{}_ys_multitags'.format(set_type)), 'r', encoding='utf8') as f:
            labels = f.read().splitlines()


        for (i, n) in enumerate(zip(txts, labels)):
            txt, label = n
            guid = "%s-%s" % (set_type, i)
            text_a=txt.split(' | ')[0]
            text_a = ' '.join(list(text_a))
            label = label.split(' | ')
            label = [' '.join(list(txt)) for txt in label]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples
