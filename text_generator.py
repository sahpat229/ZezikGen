import dataprovider
import model
import os
import tensorflow as tf

def generate_sample(dataset, n_chars_generate=600, primer=None, temperature=1.0):
    if not primer:
        primer = 'The '
    config_file_path = os.path.join("./configs/", dataset + ".json")
    training_file_path = os.path.join("./data/", dataset + ".txt")

    data_provider = dataprovider.DataProvider(training_file_path, config_file_path)
    with tf.Session() as sess:
        char_model = model.CharacterModel(data_provider, sess, config_file_path, dataset)
        char_model.initialize(restore=True)
        sampled_string = char_model.sample_model(num_chars_generate=n_chars_generate,
                                                 primer=primer,
                                                 temperature=temperature)
    return sampled_string
