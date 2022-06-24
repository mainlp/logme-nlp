#!/usr/bin/python3

import argparse, csv, json

from ud import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies Relations - Dataset Conversion')
	arg_parser.add_argument('input_path', help='path to Universal Dependencies treebank directory')
	arg_parser.add_argument('output_path', help='output prefix for corpus in HuggingFace Datasets CSV format')
	return arg_parser.parse_args()


def load_treebanks(path):
	treebanks = {}
	relations = set()

	# iterate over files in TB directory
	for tbf in sorted(os.listdir(path)):
		# skip non-conllu files
		if os.path.splitext(tbf)[1] != '.conllu': continue

		# extract treebank name (e.g. 'en-ewt-dev')
		tb_name = os.path.splitext(tbf)[0].replace('-ud-', '-').replace('_', '-')

		# load treebank
		tbf_path = os.path.join(path, tbf)
		treebank = UniversalDependencies(treebanks=[UniversalDependenciesTreebank.from_conllu(tbf_path, name=tbf)])
		relations |= set(treebank.get_relations())
		treebanks[tb_name] = treebank
		print(f"Loaded {treebank}.")

	return treebanks, sorted(relations)


def main():
	args = parse_arguments()

	# load treebank splits and relation classes
	treebanks, relations = load_treebanks(args.input_path)
	lbl_idx_map = {lbl:idx for idx, lbl in enumerate(relations)}
	print(f"Loaded {len(treebanks)} treebank splits with {len(lbl_idx_map)} relation classes.")

	# write splits to files
	for tb_name, treebank in treebanks.items():
		split_path = args.output_path + f'{tb_name}.csv'
		# write to CSV file
		with open(split_path, 'w', encoding='utf8', newline='') as output_file:
			csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
			csv_writer.writerow(['text', 'label'])
			# iterate over sentences
			for sidx in range(len(treebank)):
				# retrieve tokenized sentence and relations
				words = treebank[sidx].to_words()
				heads, rels = treebank[sidx].get_dependencies(include_subtypes=False)
				# prepare row
				text = ' '.join(words)
				labels = ' '.join([str(lbl_idx_map[rel]) for rel in rels])
				# write row to file
				csv_writer.writerow([text, labels])
		print(f"Saved {tb_name} with {sidx + 1} sentences to '{split_path}'.")

	# save relation label map
	with open(args.output_path + 'labels.json', 'w', encoding='utf8') as fp:
		json.dump(lbl_idx_map, fp, indent=4, sort_keys=True)
	print(f"Saved label map to '{args.output_path + 'labels.json'}'.")


if __name__ == '__main__':
	main()
