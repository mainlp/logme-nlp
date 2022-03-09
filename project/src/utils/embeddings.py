import sys

import numpy as np
import torch
import torch.nn as nn
import transformers


#
# Embeddings Base Class
#


class Embeddings(nn.Module):
	def __init__(self):
		super().__init__()
		self.emb_dim = None

	def __repr__(self):
		return f'<{self.__class__.__name__}: dim={self.emb_dim}>'

	def embed(self, sentences):
		"""
		Returns a list of sentence embedding matrices for list of input sentences.

		Args:
			sentences: [['t_0_0', 't_0_1, ..., 't_0_N'], ['t_1_0', 't_1_1', ..., 't_1_M']]

		Returns:
			[np.Array(sen_len, emb_dim), ...]
		"""
		raise NotImplementedError


#
# fasttext embedding model
#


class NonContextualEmbeddings(Embeddings):
	def __init__(self, word2id, embeddings, unk_token, pad_token, static=True):
		super().__init__()
		self._word2id = word2id
		self._embeddings = embeddings
		self._embeddings.requires_grad = (not static)

		# internal variables
		self._unk_token = unk_token
		self._pad_token = pad_token
		# public variables
		self.emb_dim = self._embeddings.shape[1]

	def embed(self, sentences):
		embeddings = []
		emb_words, _ = self.forward(sentences)
		for sidx, sentence in enumerate(sentences):
			cur_embeddings = emb_words[sidx, :len(sentence)].cpu().numpy()
			embeddings.append(cur_embeddings)
		return embeddings

	def forward(self, sentences):
		# sort sentences by length (max -> min)
		sentences = sorted(sentences, key=len, reverse=True)
		max_len = len(sentences[0])

		# initialize outputs
		emb_words = torch.zeros((len(sentences), max_len, self.emb_dim))
		att_words = torch.zeros((len(sentences), max_len), dtype=torch.bool)

		# iterate over sentences
		for sidx, sentence in enumerate(sentences):
			# iterate over tokens
			for tidx, token in enumerate(sentence):
				# get relevant word index or UNK index
				emb_idx = self._word2id.get(token, self._word2id[self._unk_token])
				# add to results
				emb_words[sidx, tidx] = self._embeddings[emb_idx]
				att_words[sidx, tidx] = True
			# add padding
			for tidx in range(len(sentence), max_len):
				emb_words[sidx, tidx] = self._embeddings[self._word2id[self._pad_token]]

		# move current batch of embeddings to GPU if available
		if torch.cuda.is_available():
			emb_words = emb_words.to(torch.device('cuda'))
			att_words = att_words.to(torch.device('cuda'))

		return emb_words, att_words

	@staticmethod
	def _read_vectors(file_pointer):
		# init data structures
		word2id = {}
		embeddings = []
		# iterate over embedding lines
		for eidx, line in enumerate(file_pointer):
			values = line.split(' ')
			# word is value at first position
			word = values.pop(0)
			# convert remaining values to float
			embedding = [float(v) for v in values]
			if len(embedding) != 300: print(eidx, word, len(embedding), line)

			# add to results
			word2id[word] = eidx
			embeddings.append(embedding)

			# print progress
			if eidx % 1000 == 0:
				sys.stdout.write(f"\r[Emb {eidx}] Loading embeddings...")
				sys.stdout.flush()
		print("\r", end='')
		# construct class
		return word2id, embeddings

	@staticmethod
	def from_fasttext(path, static=True):
		with open(path, 'r', encoding='utf8') as fp:
			# skip first header line
			fp.readline()
			# read vectors
			word2id, embeddings = NonContextualEmbeddings._read_vectors(fp)
			embeddings = torch.tensor(embeddings, requires_grad=(not static))
		# construct class
		return NonContextualEmbeddings(word2id, embeddings, 'UNK', 'PAD', static)

	@staticmethod
	def from_glove(path, static=True):
		with open(path, 'r', encoding='utf8') as fp:
			# read vectors
			word2id, embeddings = NonContextualEmbeddings._read_vectors(fp)
			embeddings = torch.tensor(embeddings, requires_grad=(not static))
		# construct class
		return NonContextualEmbeddings(word2id, embeddings, 'UNK', 'PAD', static)


#
# HuggingFace-based Embedding Model
#


class TransformerEmbeddings(Embeddings):
	def __init__(self, lm_name, layer=-1, cls=False, tokenized=False, static=True, special_tokens=None):
		super().__init__()
		# load tokenizer
		self._tok = transformers.AutoTokenizer.from_pretrained(lm_name, use_fast=True, add_prefix_space=True)
		# load language model
		self._lm = transformers.AutoModel.from_pretrained(lm_name, return_dict=True)
		# sanity check: some models do not have a maximum input length
		if self._tok.model_max_length != self._lm.config.max_position_embeddings:
			max_length = self._lm.config.max_position_embeddings - 2  # 2 shorter for buffer (e.g. RoBERTa)
			print(
				f"[Warning] Maximum tokenizer input length does not match language model. "
				f"Correcting {self._tok.model_max_length} to {max_length}."
			)
			self._tok.model_max_length = max_length

		# add special tokens
		if special_tokens is not None:
			self._tok.add_special_tokens({'additional_special_tokens': special_tokens})
			self._lm.resize_token_embeddings(len(self._tok))

		# move model to GPU if available
		if torch.cuda.is_available():
			self._lm.to(torch.device('cuda'))
		# set model to eval mode if used statically
		if static:
			self._lm.eval()

		# internal variables
		self._lm_layer = layer
		self._cls = cls
		self._tokenized = tokenized
		self._static = static
		# public variables
		self.emb_dim = self._lm.config.hidden_size

	def embed(self, sentences):
		embeddings = []
		emb_words, att_words = self.forward(sentences)
		# gather non-padding embeddings per sentence into list
		for sidx in range(len(sentences)):
			embeddings.append(emb_words[sidx, :len(sentences[sidx]), :].cpu().numpy())
		return embeddings

	def forward(self, sentences):
		tok_sentences = self.tokenize(sentences)
		model_inputs = {k: tok_sentences[k] for k in ['input_ids', 'token_type_ids', 'attention_mask'] if
		                k in tok_sentences}

		# perform embedding forward pass
		if self._static:
			# do not compute gradients if model is used statically
			with torch.no_grad():
				model_outputs = self._lm(**model_inputs, output_hidden_states=True)
		else:
			# retain gradients if model is not used statically
			model_outputs = self._lm(**model_inputs, output_hidden_states=True)
		# extract embeddings from relevant layer
		hidden_states = model_outputs.hidden_states  # tuple(num_layers * (batch_size, max_len, hidden_dim))
		emb_pieces = hidden_states[self._lm_layer]  # batch_size, max_len, hidden_dim

		# if input is already tokenized, reduce WordPiece to words
		if self._tokenized:
			emb_words, att_words = self.reduce(sentences, tok_sentences, emb_pieces)
			return emb_words, att_words

		# otherwise, return model-specific tokenization
		return emb_pieces, tok_sentences['attention_mask']

	def tokenize(self, sentences):
		# tokenize batch: {input_ids: [[]], token_type_ids: [[]], attention_mask: [[]], special_tokens_mask: [[]]}
		if self._tokenized:
			tok_sentences = self._tok(
				sentences,
				is_split_into_words=True, padding=True, truncation=True,
				return_tensors='pt', return_special_tokens_mask=True, return_offsets_mapping=True
			)
		else:
			tok_sentences = self._tok(
				sentences,
				padding=True, truncation=True, return_tensors='pt'
			)
		# move input to GPU (if available)
		if torch.cuda.is_available():
			tok_sentences = {k: v.to(torch.device('cuda')) for k, v in tok_sentences.items()}

		return tok_sentences

	def reduce(self, sentences, tok_sentences, emb_pieces):
		emb_words = torch.zeros_like(emb_pieces)
		att_words = torch.zeros(emb_pieces.shape[:-1], dtype=torch.bool, device=emb_pieces.device)
		max_len = 0
		# iterate over sentences
		for sidx in range(emb_pieces.shape[0]):
			# get string tokens of current sentence
			tokens = self._tok.convert_ids_to_tokens(tok_sentences['input_ids'][sidx])
			offsets = tok_sentences['offset_mapping'][sidx]

			tidx = -1
			for widx, orig_word in enumerate(sentences[sidx]):
				# init aggregate word embedding
				emb_word = torch.zeros(emb_pieces.shape[-1], device=emb_pieces.device)  # (emb_dim,)
				num_tokens = 0
				coverage = 0
				while coverage < len(orig_word):
					tidx += 1
					if tidx >= len(emb_pieces[sidx, :]):
						raise ValueError(
							f"More words than pieces {tidx} >= {len(emb_pieces[sidx, :])}.\n"
							f"UD (len={len(sentences[sidx])}): {sentences[sidx]}\n"
							f"LM (len={len(tokens)}): {tokens}"
						)
					# skip if special tokens ([CLS], [SEQ], [PAD])
					if tok_sentences['special_tokens_mask'][sidx, tidx] == 1: continue

					token_span = offsets[tidx]  # (start_idx, end_idx + 1) within orig_word
					# add WordPiece embedding to current word embedding sum
					emb_word += emb_pieces[sidx, tidx]
					num_tokens += 1
					coverage = token_span[1]

				# add mean of aggregate WordPiece embeddings and set attention to True
				emb_words[sidx, widx] = emb_word / num_tokens
				att_words[sidx, widx] = True

			# store new maximum sequence length
			max_len = len(sentences[sidx]) if len(sentences[sidx]) > max_len else max_len

		# reduce embedding and attention matrices to new maximum length
		emb_words = emb_words[:, :max_len, :]  # (batch_size, max_len, emb_dim)
		att_words = att_words[:, :max_len]  # (batch_size, max_len)

		# reattach CLS to first position
		if self._cls:
			emb_words = torch.cat((emb_pieces[:, 0:1, :], emb_words), dim=1)
			att_words = torch.cat((torch.ones((att_words.shape[0], 1), dtype=torch.bool), att_words), dim=1)

		return emb_words, att_words


#
# Pooling Functions
# (for sentence classification)
#


def get_mean_embedding(token_embeddings):
	if isinstance(token_embeddings, np.ndarray):
		return np.mean(token_embeddings, axis=0)
	elif isinstance(token_embeddings, torch.Tensor):
		return torch.mean(token_embeddings, dim=0)
	else:
		raise ValueError(f"[Error] No mean-pooling operation defined for type {type(token_embeddings)}.")


def get_first_embedding(token_embeddings):
	return token_embeddings[0]


#
# Helper Functions
#


def load_embeddings(identifier, tokenized=False, static=True, special_tokens=None):
	# embeddings from fasttext
	if identifier.startswith('fasttext:'):
		vector_file = identifier.split(':')[1]
		return NonContextualEmbeddings.from_fasttext(vector_file, static=static)
	# embeddings from GloVe
	if identifier.startswith('glove:'):
		vector_file = identifier.split(':')[1]
		return NonContextualEmbeddings.from_glove(vector_file, static=static)
	# embeddings from pre-trained transformer model
	if identifier.startswith('transformer:'):
		lm_name = identifier.split(':')[1]
		transformers.logging.set_verbosity_error()
		return TransformerEmbeddings(lm_name, tokenized=tokenized, static=static, special_tokens=special_tokens)
	# embeddings + CLS-token from pre-trained transformer model
	if identifier.startswith('transformer+cls:'):
		lm_name = identifier.split(':')[1]
		transformers.logging.set_verbosity_error()
		return TransformerEmbeddings(lm_name, cls=True, tokenized=tokenized, static=static, special_tokens=special_tokens)
	else:
		raise ValueError(f"[Error] Unknown embedding specification '{identifier}'.")


def load_pooling_function(identifier):
	if identifier == 'mean':
		return get_mean_embedding
	elif identifier == 'first':
		return get_first_embedding
	else:
		raise ValueError(f"[Error] Unknown pooling specification '{identifier}'.")
