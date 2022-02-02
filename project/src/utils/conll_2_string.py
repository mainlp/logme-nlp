import argparse
import json
import csv

def bio_to_csv(inputfile, outputfile, labelsfile):

    with open(labelsfile) as file:
        label_dict = json.load(file)

    with open(inputfile) as input:
        tokens, labels = [], []
        instance_tokens, instance_labels = '', ''

        for line in input:
            if line != '\n':
                t = line.strip().split('\t')[0]
                instance_tokens += f' {t}'
                l = line.strip().split('\t')[1]
                instance_labels += f' {label_dict[l]}'
            else:
                tokens.append(instance_tokens.strip())
                labels.append(instance_labels.strip())
                instance_tokens, instance_labels = '', ''
        tokens.append(instance_tokens.strip())
        labels.append(instance_labels.strip())

    with open(outputfile, 'w', encoding='utf8', newline='') as output:
        csv_writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        csv_writer.writerow(['text', 'label'])
        for t, l in zip(tokens, labels):
            csv_writer.writerow([t, l])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Convert data from bio format to csv format.")
    parser.add_argument("--input", help="Input file in the conll format.")
    parser.add_argument("--output", help="Output file in the csv HuggingFace format.")
    parser.add_argument("--labels", help="JSON file with the labels.")
    args = parser.parse_args()

    bio_to_csv(args.input, args.output, args.labels)