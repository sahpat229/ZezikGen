import json
import numpy as np


class DataProvider():
    def __init__(self, training_file_path, config_file_path):
        with open(training_file_path) as file:
            text = file.read()

        with open(config_file_path) as config_file:
            self.config = json.load(config_file)

        unique_chars = set(text)
        self.vocab_size = len(set(text))
        self.char_to_ix = {char: i for i, char in enumerate(unique_chars)}
        self.ix_to_char = {i: char for i, char in enumerate(unique_chars)}

        self.np_text = self.process_text(text)
        self.create_batches()
        self.batch_number = 0

    def convert_char(self, char):
        return self.char_to_ix[char]

    def convert_char_index(self, char_index):
        return self.ix_to_char[char_index]

    def create_batches(self):
        if self.np_text.size - 1 < (self.config['batch_size'] * self.config['timesteps']):
            raise ValueError('Decrease batch_size or timesteps, not enough data')

        self.num_batches = int((self.np_text.size - 1) / (self.config['batch_size'] * self.config['timesteps']))
        print("text len:", self.np_text.size)
        print("NUM Batches:", self.num_batches)
        x_text = self.np_text[0:self.num_batches * self.config['batch_size'] * self.config['timesteps']]
        y_text = self.np_text[1:self.num_batches * self.config['batch_size'] * self.config['timesteps'] + 1]

        self.x_batches = np.reshape(x_text, [self.num_batches, self.config['batch_size'], self.config['timesteps']])
        self.y_batches = np.reshape(y_text, [self.num_batches, self.config['batch_size'], self.config['timesteps']])

    def process_text(self, text):
        return np.array(list(map(self.char_to_ix.get, text)))

    def sample_batch(self):
        reset = False
        if self.batch_number >= self.num_batches:
            self.batch_number = 0
            reset = True

        x, y = self.x_batches[self.batch_number], self.y_batches[self.batch_number]
        self.batch_number += 1

        return x, y, reset
