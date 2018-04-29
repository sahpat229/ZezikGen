import argparse
import dataprovider
import model
import os
import tensorflow as tf

parser = argparse.ArgumentParser(description="Train the LSTM model")
parser.add_argument('--dataset', type=str, required=True)
args = vars(parser.parse_args())
dataset = args['dataset']

config_file_path = os.path.join("./configs/", dataset + ".json")
training_file_path = os.path.join("./data/", dataset + ".txt")

data_provider = dataprovider.DataProvider(training_file_path, config_file_path)

sess = tf.Session()
char_model = model.CharacterModel(data_provider, sess, config_file_path, dataset)
char_model.initialize(restore=True)
char_model.train()
