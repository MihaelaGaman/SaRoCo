import pandas as pd
import json
import csv

from sklearn.utils import shuffle

# Custom
from splits import Data
from model import CharCNN

if __name__ == "__main__":
    # Paths
    base_path = "../../data/"
    train_path = base_path + "train.csv"
    test_path = base_path + "test.csv"
    val_path = base_path + "validation.csv"

    # Load the data
    df_train = pd.read_csv(train_path, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    df_val = pd.read_csv(val_path, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    df_test = pd.read_csv(test_path, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Preprocess a little
    df_train = df_train.replace('', 'null').replace(' ', 'null').dropna(subset=["content"])
    df_train = shuffle(df_train)

    df_val = df_val.replace('', 'null').replace(' ', 'null').dropna(subset=["content"])
    df_val = shuffle(df_val)

    df_test = df_test.replace('', 'null').replace(' ', 'null').dropna(subset=["content"])
    df_test = shuffle(df_test)

    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    df_val = df_val.reset_index()

    # Find out the data splits sizes
    print('Number of training sentences: {:,}\n'.format(df_train.shape[0]))
    print('Number of validation sentences: {:,}\n'.format(df_val.shape[0]))
    print('Number of test sentences: {:,}\n'.format(df_test.shape[0]))

    """ Prepare the data for classification """
    # Get the lists of contents and their labels.
    contents_train = df_train.content.values
    labels_train = df_train.label.values

    contents_val = df_val.content.values
    labels_val = df_val.label.values

    contents_test = df_test.content.values
    labels_test = df_test.label.values

    """ Model """
    # Load config
    config_path = "./config.json"
    config = json.load(open(config_path, encoding="utf-8", errors="ignore"))

    # Variables useful in further data processing
    alphabet = config["data"]["alphabet"]
    input_size = config["data"]["input_size"]
    number_of_classes = 2

    # Make conversions (featurize and convert labels to one-hot vectors)
    dataTrain = Data(list(zip(contents_train, labels_train)), alphabet, input_size)
    train_data, train_labels = dataTrain.convert_data()

    dataVal = Data(list(zip(contents_val, labels_val)), alphabet, input_size)
    val_data, val_labels = dataVal.convert_data()

    dataTest = Data(list(zip(contents_test, labels_test)), alphabet, input_size)
    test_data, test_labels = dataTest.convert_data()

    # Initialize the model
    model = CharCNN(
        input_sz=config["data"]["input_size"],
        alphabet_sz=config["data"]["alphabet_size"],
        emb_sz=config["char_cnn_zhang"]["embedding_size"],
        conv_layers=config["char_cnn_zhang"]["conv_layers"],
        fc_layers=[],
        threshold=config["char_cnn_zhang"]["threshold"],
        dropout_p=config["char_cnn_zhang"]["dropout_p"],
        optimizer=config["char_cnn_zhang"]["optimizer"],
        loss=config["char_cnn_zhang"]["loss"]
    )

    # Train
    model.train(
        train_inputs=train_data,
        train_labels=train_labels,
        val_inputs=val_data,
        val_labels=val_labels,
        epochs=config["training"]["epochs"],
        bs=config["training"]["batch_size"]
    )

    # Evaluate
    results = model.test(test_data, test_labels, bs=128)
    model.test_model(test_data, test_labels, bs=128)
    print(results)

    # Save the model
    model.model.save('model.h5')
