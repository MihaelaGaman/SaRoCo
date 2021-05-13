import numpy as np


class Data(object):
    """
    Class to handle the conversions of characters to indexes.
    """
    def __init__(self, dataset, alphabet, input_size):
        """
        Initialize a Data object.
        :param dataset: Raw dataset.
        :param alphabet: Alphabet to be used.
        :param input_size: Input window considered.
        """
        # Initialize the alphabet
        self.alphabet = alphabet
        # Number of classes
        self.no_of_classes = 2
        # Input size
        self.length = input_size
        # Data set initialization
        self.dataset = dataset

        # Map each character to an integer
        self.dict = {}
        for idx, char in enumerate(alphabet):
            self.dict[char] = idx + 1

    def convert_data(self):
        """
        Convert the inputs in numeric format and the labels into one-hot encoding.
        :return: Data transformed from raw to indexed form with associated one-hot label.
        """
        one_hot = np.eye(self.no_of_classes, dtype='int16')
        classes = []
        batch_indices = []
        for text, label in self.dataset:
            batch_indices.append(self.str_to_indexes(text))
            classes.append(one_hot[int(label)])
        return np.asarray(batch_indices, dtype="int16"), np.asarray(classes)

    def str_to_indexes(self, text):
        """
        Convert a string to character indexes based on character dictionary.
        :param text: String to be converted to indexes.
        :return: Np array of indexes of the characters in text.
        """
        max_length = min(len(text), self.length)
        str2idx = np.zeros(self.length, dtype="int16")
        for i in range(1, max_length + 1):
            char = text[-i]
            if char in self.dict:
                str2idx[i - 1] = self.dict[char]
        return str2idx

