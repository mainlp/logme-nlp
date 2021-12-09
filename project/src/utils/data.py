import json
import re

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
            # slice data
            inputs = self._inputs[start_idx:end_idx]
            labels = self._labels[start_idx:end_idx]
            # yield batch
            yield inputs, labels

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

    @staticmethod
    def from_path(path):
        inputs, labels = [], []
        instance_pattern = re.compile(
                r'^(?P<inputs>\[.+?\])\s(?P<labels>".+?"|\[.+?\])'
                )
        with open(path, 'r', encoding='utf8') as fp:
            for lidx, line in enumerate(fp):
                line = line.strip()
                instance_match = instance_pattern.match(line)
                # check: skip lines with non-conforming format
                if instance_match is None:
                    raise ValueError(
                            f"[Error] Line {lidx} of '{path}' does not follow the dataset specification format.")
                # parse input sequences and labels
                instance_inputs = json.loads(instance_match['inputs'])
                instance_labels = json.loads(instance_match['labels'])
                # check: input has one label or the same number of labels as sequence items
                if (type(instance_labels) is list) and (len(instance_inputs) != len(instance_labels)):
                    raise ValueError(
                            f"[Error] Line {lidx} of '{path}' has an unequal number of sequence items and labels.")
                # append inputs and labels to overall dataset
                inputs.append(instance_inputs)
                labels.append(instance_labels)

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
