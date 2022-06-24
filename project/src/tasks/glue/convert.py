#!/usr/bin/python3

import argparse, csv, json, os

from datasets import load_dataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='GLUE - Dataset Conversion')
	arg_parser.add_argument('tasks', nargs='+', help='list of GLUE tasks to convert')
	arg_parser.add_argument('output_path', help='output prefix for corpus in HuggingFace Datasets CSV format')
	arg_parser.add_argument('-s', '--sep_token', default=' ', help='separator token to use for multi-sentence tasks')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	# load GLUE subtask from HF Datasets
	lbl_idx_map = {'unknown': -1}
	for task in args.tasks:
		glue_data = load_dataset('glue', task)
		print(f"Loaded GLUE dataset '{task}' with splits {', '.join(glue_data.keys())}.")

		for split in glue_data:
			columns = list(glue_data[split].features.keys())

			# extract labels
			labels = glue_data[split]['label']
			lbl_idx_map.update({lbl: idx for idx, lbl in enumerate(glue_data[split].features['label'].names)})

			# check for single-sentence tasks (i.e., label is second column)
			if columns.index('label') == 1:
				texts = glue_data[split][columns[0]]
			# check for multi-sentence tasks (i.e., label is third column)
			else:
				texts1 = glue_data[split][columns[0]]
				texts2 = glue_data[split][columns[1]]
				# concatenate sentences based on provided SEP token
				texts = [t1 + args.sep_token + t2 for t1, t2 in zip(texts1, texts2)]

			assert len(texts) == len(labels), f"[Error] Number of texts and labels does not match ({len(texts)} != {len(labels)})."

			# write to CSV output
			split_path = os.path.join(args.output_path, f'{task}-{split}.csv')
			with open(split_path, 'w', encoding='utf8', newline='') as output_file:
				csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
				csv_writer.writerow(['text', 'label'])
				csv_writer.writerows(zip(texts, labels))
			print(f"Saved {task}-{split} with {len(texts)} sentences to '{split_path}'.")

		# save relation label map
		map_path = os.path.join(args.output_path, f'{task}-labels.json')
		with open(map_path, 'w', encoding='utf8') as fp:
			json.dump(lbl_idx_map, fp, indent=4, sort_keys=True)
		print(f"Saved label map to '{map_path}'.")


if __name__ == '__main__':
	main()
