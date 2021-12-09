import torch
import torch.nn as nn

#
# Loss Functions
#


class LabelLoss(nn.Module):
	def __init__(self, classes):
		super().__init__()
		self._xe_loss = nn.CrossEntropyLoss(ignore_index=-1)
		self._classes = classes
		self._class2id = {c:i for i, c in enumerate(self._classes)}

	def __repr__(self):
		return f'<{self.__class__.__name__}: loss=XEnt, num_classes={len(self._classes)}>'

	def forward(self, logits, targets):
		# map target label strings to class indices
		target_labels = torch.tensor(
			[self._class2id[c] for c in targets], dtype=torch.long,
			device=logits.device
		)

		return self._xe_loss(logits, target_labels)

	def get_accuracy(self, logits, targets):
		# map target label strings to class indices
		target_labels = torch.tensor(
			[self._class2id[c] for c in targets], dtype=torch.long,
			device=logits.device
		)
		# get labels from logits
		labels = torch.argmax(logits, dim=-1)

		# compute label accuracy
		num_label_matches = torch.sum(labels == target_labels)
		accuracy = float(num_label_matches / labels.shape[0])

		return accuracy