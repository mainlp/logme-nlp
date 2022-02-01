import argparse

def bio_to_csv(inputfile, outputfile):

    output = open(outputfile, 'w')
    output.write(f'"text","label"\n')

    with open(inputfile) as input:

        tokens, labels = '', ''

        for line in input:
            if line != '\n':
                t = line.strip().split('\t')[0]
                tokens += f' {t}'
                l = line.strip().split('\t')[1]
                labels += f' {l}'
            else:
                tokens = '"' + tokens.strip() + '"'
                labels = '"' + labels.strip() + '"'
                output.write(f'{tokens},{labels}\n')
                tokens, labels = '', ''

    output.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Convert data from bio format to csv format.")
    parser.add_argument("--input", help="Input file in the conll format.")
    parser.add_argument("--output", help="Output file n the csv HuggingFace format.")
    args = parser.parse_args()

    bio_to_csv(args.input, args.output)