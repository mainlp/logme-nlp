from .classifiers import *
from .losses import *


def load_classifier(identifier):
	if identifier == 'linear':
		return LinearClassifier, LabelLoss
	elif identifier == 'mlp':
		return MultiLayerPerceptronClassifier, LabelLoss
	else:
		raise ValueError(f"[Error] Unknown classifier specification '{identifier}'.")
