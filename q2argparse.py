# Import the library
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()
print('Hello,', args.name, args.seed)