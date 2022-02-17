import argparse
import json
import csv

def csv_to_bio(inputfile, outputfile, dict_labels):

    output = open(outputfile, 'w')

    with open(inputfile) as input:
        csvreader = csv.reader(input)
        header = next(csvreader)
        for row in csvreader:
            tokens = row[0].split(' ')
            labels = row[1].split(' ')
            for token, label in zip(tokens, labels):
                output.write(f'{token}\t{dict_labels[int(label)]}\n')

    output.close()


def revert_dict(labelsfile):

    with open(labelsfile) as file:
        line = file.readline()
        old_labels = json.loads(line)
    reverted_labels = {id:label for label, id in old_labels.items()}


    return reverted_labels

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Convert data from bio format to csv format.")
    parser.add_argument("--input", help="Input file in the csv HuggingFace format.") #
    parser.add_argument("--output", help="Output file in the conll format.")
    parser.add_argument("--labels", help="JSON file with the labels.")
    args = parser.parse_args()

    labels = revert_dict(args.labels)

    csv_to_bio(args.input, args.output, labels)