from code.data_prep import load_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_folder', help='input folder', action='store', required=True)
parser.add_argument('-e', '--extra_folder', default=None, help='extra folder', action='store')

opt = parser.parse_known_args()[0]

TRAIN_DATA = opt.input_folder + '/'
EXTRA_DATA = opt.extra_folder

load_data(TRAIN_DATA, EXTRA_DATA)
