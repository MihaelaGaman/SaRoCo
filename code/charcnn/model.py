from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Multiply, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, BatchNormalization
from tensorflow.keras.layers import ThresholdedReLU, Dropout, GlobalAveragePooling1D, AlphaDropout
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from metrics import Metrics, compute_print_f1


class CharCNN(object):
    """
    Class that implements the Character Level Convolutional Neural Network for Text Classification,
    as described in Zhang et al., 2015 (http://arxiv.org/abs/1509.01626) with the Squeeze-and-Excitation
    blocks from Butnaru and Ionescu, 2019 (https://arxiv.org/abs/1901.06543)
    """
    def __init__(self, input_sz, alphabet_sz, emb_sz,
                 conv_layers, fc_layers,
                 threshold, dropout_p,
                 optimizer='adam', loss='categorical_crossentropy'):
        """
        Initialize the Character Level CNN model.
        :param input_sz: Input window considered.
        :param alphabet_sz: Alphabet size.
        :param emb_sz: Embedding vector sizes.
        :param conv_layers: The list of convolutional layers properties.
        :param fc_layers: The list of fully connected layers to be used.
        :param threshold: Threshold for the Thresholded ReLU activation function.
        :param dropout_p: Dropout rate.
        :param optimizer: Optimization algorithm.
        :param loss: Loss function.
        """
        # Initialize the parameters
        self.input_sz = input_sz
        self.alphabet_sz = alphabet_sz
        self.emb_sz = emb_sz
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.num_of_classes = 2
        self.threshold = threshold
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.loss = loss
        # Build and compile the model
        self._build_model()

    def _squeeze_and_excitation_block(self, input_data: np.ndarray, ratio: int) -> Layer:
        """
        Squeeze and excitation block implementation.
        :param input_data: Input data.
        :param ratio: Squeeze ratio.
        :return: 
        """
        out_dim = int(input_data.shape[-1])

        # Squeeze step
        squeeze = GlobalAveragePooling1D()(input_data)
        squeeze = Reshape((-1, out_dim))(squeeze)

        # Excitation
        excitation = Dense(
            int(out_dim / ratio),
            activation="relu",
            kernel_initializer="he_normal",
            use_bias=False
        )(squeeze)
        excitation = Dense(
            out_dim,
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=False
        )(excitation)

        # Scale
        scale = Multiply()([input_data, excitation])

        return scale

    def _build_model(self) -> None:
        """
        Build and compile the Character Level CNN model
        :return: None
        """
        # Input layer
        inputs = Input(shape=(self.input_sz,), name='sent_input', dtype='int16')

        # Embedding layers
        x = Embedding(self.alphabet_sz + 1, self.emb_sz, input_length=self.input_sz)(inputs)
        x = Reshape((self.input_sz, self.emb_sz))(x)

        # Convolutional layers
        for cl in self.conv_layers:
            x = Conv1D(cl[0], cl[1], kernel_initializer="lecun_normal", padding="causal", use_bias=False)(x)
            x = BatchNormalization(scale=False)(x)
            x = Activation('selu')(x)
            x = AlphaDropout(0.5)(x)

            if cl[2] != -1:
                x = MaxPooling1D(cl[2], cl[3])(x)
            if cl[4] != -1:
                x = self._squeeze_and_excitation_block(input_data = x, ratio = cl[4])

        # Flatten the features
        x = Flatten()(x)

        # Fully connected layers
        for fl in self.fc_layers:
            x = Dense(fl)(x)
            x = ThresholdedReLU(self.threshold)(x)
            x = Dropout(self.dropout_p)(x)

        # Output layer
        predictions = Dense(self.num_of_classes, activation="softmax")(x)

        # Build and coompile the model
        model = Model(inputs, predictions)

        # Compile
        model.compile(optimizer='nadam', loss=self.loss, metrics=['accuracy'])
        self.model = model
        self.model.summary()

    def train(self, train_inputs, train_labels, val_inputs, val_labels, epochs, bs):
        """
        Train the model.
        :param train_inputs: Training data.
        :param train_labels: Training labels.
        :param val_inputs: Validation data.
        :param val_labels: Validation labels.
        :param epochs: Epochs.
        :param bs: Batch size.
        :return: None
        """
        # Create callbacks
        filepath= "checkpoints/weights-improvement-{epoch:02d}-acc-{val_accuracy:.2f}-loss-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        metrics = Metrics((val_inputs, val_labels))
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)

        callbacks_list = [checkpoint, metrics, es]

        self.model.fit(
            train_inputs, train_labels,
           validation_data=(val_inputs, val_labels),
           epochs=epochs,
           batch_size=bs,
           verbose=1,
           callbacks=callbacks_list
        )

    def test(self, test_inputs, test_labels, bs):
        """
        Evaluate the model.
        :param test_inputs: Test data.
        :param test_labels: Test labels.
        :param bs: Batch size.
        :return: Results / Metrics.
        """
        # Evaluate inputs
        results = self.model.evaluate(test_inputs, test_labels, batch_size=bs, verbose=1)
        return results

    def test_model(self, test_inputs, test_labels, bs):
        """
        Evaluate the model and display results.
        :param test_inputs: Test data.
        :param test_labels: Test labels.
        :param bs: Batch size.
        :return: None
        """
        # Use callbacks
        metrics = Metrics((test_inputs, test_labels))

        # Evaluate inputs
        eval_res =  self.model.evaluate(test_inputs, test_labels,
                                        batch_size=bs,
                                        callbacks=[metrics])
        print(eval_res)

        labels = []
        for labels_arr in test_labels:
            for i in range(len(labels_arr)):
                if labels_arr[i] == 1:
                    labels.append(i)

        # Predict
        predicts = self.model.predict(test_inputs, batch_size=bs, verbose=1)
        pred_arr = np.argmax(predicts, axis=1)

        # F1 score weighted
        compute_print_f1(pred_arr, np.asarray(labels), "weighted")

        # F1 score macro
        compute_print_f1(pred_arr, np.asarray(labels), "macro")

    def save(self, file_path):
        """
        Save the model
        :param file_path: Path where the model will be saved.
        :return: None
        """
        self.model.save(file_path)

