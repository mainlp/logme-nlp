import csv
import json
import sys

import numpy as np


#
# LEEP Input Dataset
#


class LabelledDataset:
    def __init__(self, inputs, labels):
        self._inputs = inputs  # List(List(Str)): [['t0', 't1', ...], ['t0', 't1', ...]] or List(Str): ['t0 t1 ... tN']
        self._labels = labels  # List(List(Str)): [['l0', 'l1', ...], ['l0', 'l1', ...]] or List(Str): ['l0', 'l1', ...]

    def __len__(self):
        return len(list(self.get_flattened_labels()))

    def __repr__(self):
        return f'<LabelledDataset: {len(self._inputs)} inputs, {len(self)} labels>'

    def get_flattened_labels(self):
        for cur_labels in self._labels:
            if type(cur_labels) is list:
                for cur_label in cur_labels:
                    yield cur_label
            else:
                yield cur_labels

    def get_label_types(self):
        label_types = set()
        for label in self.get_flattened_labels():
            label_types.add(label)
        return sorted(label_types)

    def get_batches(self, batch_size):
        cursor = 0
        while cursor < len(self._inputs):
            # set up batch range
            start_idx = cursor
            end_idx = min(start_idx + batch_size, len(self._inputs))
            cursor = end_idx
            num_remaining = len(self._inputs) - cursor - 1
            # slice data
            inputs = self._inputs[start_idx:end_idx]
            labels = self._labels[start_idx:end_idx]
            # flatten sequential labels if necessary
            if type(labels[0]) is list:
                labels = [l for seq in labels for l in seq]
            # yield batch
            yield inputs, labels, num_remaining

    def get_shuffled_batches(self, batch_size):
        # start with list of all input indices
        remaining_idcs = list(range(len(self._inputs)))
        np.random.shuffle(remaining_idcs)

        # generate batches while indices remain
        while len(remaining_idcs) > 0:
            # pop-off relevant number of instances from pre-shuffled set of remaining indices
            batch_idcs = [remaining_idcs.pop() for _ in range(min(batch_size, len(remaining_idcs)))]

            # gather batch data
            inputs = [self._inputs[idx] for idx in batch_idcs]
            # flatten sequential labels if necessary
            if type(self._labels[batch_idcs[0]]) is list:
                labels = [l for idx in batch_idcs for l in self._labels[idx]]
            # one label per input does not require flattening
            else:
                labels = [self._labels[idx] for idx in batch_idcs]
            # yield batch + number of remaining instances
            yield inputs, labels, len(remaining_idcs)

    def save(self, path):
        with open(path, 'w', encoding='utf8', newline='') as output_file:
            csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
            csv_writer.writerow(['text', 'label'])
            for idx, text in enumerate(self._inputs):
                text = ' '.join(text)
                label = self._labels[idx]
                if type(label) is list:
                    label = ' '.join([str(l) for l in label])
                csv_writer.writerow([text, label])

    @staticmethod
    def from_path(path):
        inputs, labels = [], []
        label_level = 'sequence'
        with open(path, 'r', encoding='utf8', newline='') as fp:
            csv.field_size_limit(sys.maxsize)
            csv_reader = csv.DictReader(fp)
            for row in csv_reader:
                # convert all previous labels to token-level when encountering the first token-level label set
                if (' ' in row['label']) and (label_level != 'token'):
                    labels = [[l] for l in labels]
                    label_level = 'token'
                # covert current label(s) into appropriate form
                if label_level == 'token':
                    label = row['label'].split(' ')
                else:
                    label = row['label']
                # append inputs and labels to overall dataset
                inputs.append(row['text'])
                labels.append(label)

        return LabelledDataset(inputs, labels)


#
# LEEP Output Dataset
#


class LeepWriter:
    def __init__(self, path):
        self._file_pointer = open(path, 'w', encoding='utf8')

    @staticmethod
    def _convert_to_leep(probabilities, label):
        return f'{[float(p) for p in probabilities]} "{label}"\n'

    def write(self, data):
        self._file_pointer.write(data)

    def close(self):
        self._file_pointer.close()

    def write_header(self, source_labels, target_labels):
        source_labels_str = f'# {json.dumps(source_labels, ensure_ascii=False)}\n'
        target_labels_str = f'# {json.dumps(target_labels, ensure_ascii=False)}\n'
        self.write(target_labels_str + source_labels_str)

    def write_instance(self, source_probabilities, target_label):
        self.write(self._convert_to_leep(source_probabilities, target_label))

    def write_instances(self, source_probabilities, target_labels):
        output = ''
        for idx in range(source_probabilities.shape[0]):
            output += self._convert_to_leep(source_probabilities[idx], target_labels[idx])
        self.write(output)
