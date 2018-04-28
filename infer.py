import dataprovider
import model
import tensorflow as tf

config_file_path = "./config.json"
training_file_path = "./data/zezik.txt"

data_provider = dataprovider.DataProvider(training_file_path, config_file_path)

sess = tf.Session()
char_model = model.CharacterModel(data_provider, sess, config_file_path)
char_model.restore()
char_model.infer()
