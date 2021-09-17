import argparse, itertools
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


class LogExpectedEmpiricalPrediction:
    """Class to calculate the leep score from a .txt file consisting of structure:
    # target labels Y
    # source labels Z
    <output probabilities of pretrained model theta on Z applied on Y> <gold label of y>
    example:
    # [A, B, C, D]
    # [U, V, W, X, Y, Z]
    [0.1, 0.2, 0.3, 0.1, 0.1, 0.2] A
    ...
    """

    def __init__(self, path: str):
        self.path = path
        self._label_index_source = defaultdict(int)
        self._label_index_target = defaultdict(int)

    def _create_label_indices(self, target_labels: List[Any], source_labels: List[Any]) -> None:
        """Creates a label index of both target labels and source labels

        :param target_labels: list of unique gold labels of target set Y
        :type target_labels: List[Any]
        :param source_labels: list of unique gold labels of source set Z
        :type source_labels: List[Any]

        :rtype: None
        """
        cnt_target = 0
        cnt_source = 0

        for label_target in target_labels:
            self._label_index_target[label_target] += cnt_target
            cnt_target += 1

        for label_source in source_labels:
            self._label_index_source[label_source] += cnt_source
            cnt_source += 1

    def _read_data(self) -> Tuple[List[Any], List[Any], List[Tuple[List, Any]]]:
        """Compute the empirical conditional distribution P_hat(y|z) of the target label y given the source label z

        :rtype: Tuple[List[Any], List[Any], List[Tuple[List, Any]]]
        :return: returns the unique target labels, gold labels and the output probabilities with corresponding gold
        label from a .txt file indicated by a path.
        """
        with open(self.path) as f:
            data = f.readlines()
            target_labels = data[0].strip("# ][ \n").split(", ")
            source_labels = data[1].strip("# ][ \n").split(", ")
            self._create_label_indices(target_labels, source_labels)

            output_probabilities = []

            for line in data[2:]:
                line = line.strip()
                gold_label = line.split("]")[-1].strip()
                dummy_probas = [float(proba) for proba in line.strip()[:-len(gold_label)].strip("][ \n").split(", ")]
                output_probabilities.append((dummy_probas, gold_label))

        return target_labels, source_labels, output_probabilities

    def _get_cartesian_product_labels(self, target_labels: List[Any], source_labels: List[Any]) -> List[Tuple]:
        """Creates all possible label combinations from target labels and source labels

        :param target_labels: list of unique gold labels of target set Y
        :type target_labels: List[Any]
        :param source_labels: list of unique gold labels of source set Z
        :type source_labels: List[Any]

        :rtype: List[Tuple[Any, Any]]
        :return: cartesian product of target labels and source labels
        """
        target_source_product = list(itertools.product(target_labels, source_labels))

        return target_source_product

    def _compute_joint_distribution(self,
                                    target_source_product: List[Tuple],
                                    output_probabilities: List[Tuple[List, Any]]) -> Dict[Tuple, float]:
        """Compute the empirical joint distribution P_hat(y,z) of the target label y and the source label z

        :param target_source_product: all possible label combinations of gold labels of source and target set.
        :type target_source_product: List[Tuple[Any, Any]]
        :param output_probabilities: a list of tuples consisting of the output probabilities of a pretrained model theta
        trained on source set Z and applied on target set Y with label distribution from Z.
        :type output_probabilities: List[Tuple[List, Any]]

        :rtype: Dict[Tuple, float]
        :return: defaultdict object with joint probabilities of each combination of (y,z) with tuple of label
        combination as key and the joint probability as value.
        """

        joint_distribution = defaultdict(int)

        for (target_label, source_label) in target_source_product:
            dummy_output_probabilities = []

            for output_probability, gold_target_label in output_probabilities:

                if self._label_index_target[target_label] == self._label_index_target[gold_target_label]:
                    dummy_output_probabilities.append(output_probability[self._label_index_source[source_label]])

            # p(y,z) = (sum_{y=Y} theta(x_i)_z) / n
            joint_probability = sum(dummy_output_probabilities) / len(output_probabilities)
            joint_distribution[(target_label, source_label)] = joint_probability

        return joint_distribution

    def _compute_conditional_distribution(self, joint_distribution: Dict[Tuple, float]) -> Dict[Tuple, float]:
        """Compute the empirical conditional distribution P_hat(y|z) of the target label y given the source label z

        :param joint_distribution: joint probabilities of each combination of (y,z) with tuple of label combination
        as key and the joint probability as value.
        :type output_probabilities: Dict[Tuple, float]

        :rtype: Dict[Tuple, float]
        :return: defaultdict object with tuple of labels from cartesian product source Z and target Y as key,
        conditional distribution as value
        """
        ### marginal distribution of z ###

        marginal = defaultdict(list)

        for (target_label, source_label), joint_probability in joint_distribution.items():
            marginal[target_label].append(joint_probability)

        ### conditional distribution of p(y|z) ###

        conditional_distribution = defaultdict(int)

        for cls, joint_probabilities in marginal.items():
            marginal_z = sum(joint_probabilities)  # compute marginal of z here: p(z) = sum_y_in_Y p(y,z)

            for (target_label, source_label), joint_probability in joint_distribution.items():

                if self._label_index_target[cls] == self._label_index_source[source_label]:
                    if marginal_z > 0:
                        # p(y|z) = p(y,z)/p(z)
                        conditional_distribution[(target_label, source_label)] = joint_probability / marginal_z
                    else:
                        conditional_distribution[(target_label, source_label)] = 0.

        return conditional_distribution

    def compute_leep(self) -> float:
        """Compute LEEP measure

        :rtype: float
        :return: LEEP measure
        """

        target_labels, source_labels, output_probabilities = self._read_data()
        target_source_product = self._get_cartesian_product_labels(target_labels, source_labels)
        joint_distribution = self._compute_joint_distribution(target_source_product, output_probabilities)
        conditional_distribution = self._compute_conditional_distribution(joint_distribution)
        leep = 0

        for output_probability, target_label in output_probabilities:
            eep = 0

            for (_, z_target), conditional in conditional_distribution.items():
                if self._label_index_source[z_target] != self._label_index_target[target_label]:
                    continue

                eep += conditional * output_probability[self._label_index_source[z_target]]

            if eep != 0:
                leep += np.log(eep)

        leep /= len(output_probabilities)

        return leep


def main():
    arg_parser = argparse.ArgumentParser(description='Log Expected Empirical Prediction (LEEP)')
    arg_parser.add_argument('predictions', help='path to text file with prediction probabilities')
    args = arg_parser.parse_args()

    leep = LogExpectedEmpiricalPrediction(args.predictions)
    print(f"LEEP: {leep.compute_leep()}")


if __name__ == '__main__':
    main()
