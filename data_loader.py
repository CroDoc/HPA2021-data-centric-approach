from code.data_prep import load_data
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_folder', help='input folder', action='store')
parser.add_argument('-e', '--extra_folder', help='extra folder', action='store')
args = vars(parser.parse_args())

if not args['input_folder']:
    raise Exception('input folder needs to be specified')

TRAIN_DATA = args['input_folder'] + '/'

if args['extra_folder']:
    EXTRA_DATA = args['extra_folder']
else:
    EXTRA_DATA = None

load_data(TRAIN_DATA, EXTRA_DATA)