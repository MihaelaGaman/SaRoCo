from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


def compute_print_f1(predict, targ, average):
    """
    Compute the F1 score achieved at prediction, given the target labels and
    the averaging scheme preferred (i.e. macro or weighted).
    :param predict:
    :param targ:
    :param average:
    :return:
    """
    f1 = f1_score(targ, predict, average=average)
    recall = recall_score(targ, predict, average=average)
    precision = precision_score(targ, predict, average=average)

    print("[%s] test_f1: %f \t test_precision: %f \t test_recall %f"
          % (average, f1, precision, recall))

    return f1, recall, precision


class Metrics(Callback):
    """
    Implement the Metrics class to be used in the callbacks list.
    """
    def __init__(self, val_data, batch_size=128):
        super().__init__()
        self.validation_data = val_data

    def on_train_begin(self, logs={}):
        self.val_f1s_weighted = []
        self.val_recalls_weighted = []
        self.val_precisions_weighted = []

        self.val_f1s_macro = []
        self.val_recalls_macro = []
        self.val_precisions_macro = []

    def evaluate_f1s(self, val_predict, val_targ):
        _val_f1_weighted = f1_score(val_targ, val_predict, average='weighted')
        _val_recall_weighted = recall_score(val_targ, val_predict, average='weighted')
        _val_precision_weighted = precision_score(val_targ, val_predict, average='weighted')
        self.val_f1s_weighted.append(_val_f1_weighted)
        self.val_recalls_weighted.append(_val_recall_weighted)
        self.val_precisions_weighted.append(_val_precision_weighted)

        _val_f1_macro = f1_score(val_targ, val_predict, average='macro')
        _val_recall_macro = recall_score(val_targ, val_predict, average='macro')
        _val_precision_macro = precision_score(val_targ, val_predict, average='macro')
        self.val_f1s_macro.append(_val_f1_macro)
        self.val_recalls_macro.append(_val_recall_macro)
        self.val_precisions_macro.append(_val_precision_macro)

        print("[weighted] val_f1: %f val_precision: %f  val_recall %f" % (_val_f1_weighted, _val_precision_weighted, _val_recall_weighted))
        print("[MACRO] val_f1: %f val_precision: %f  val_recall %f" % (_val_f1_macro, _val_precision_macro, _val_recall_macro))
        return

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]

        self.evaluate_f1s(val_predict, val_targ)
        return
