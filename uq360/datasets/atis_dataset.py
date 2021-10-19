import csv
import os

import numpy as np
import sys


def default_preprocessing(data):
    """
    Grab query text (features) and class intent (labels).
    """

    # create training data and labels
    x = []
    y = []
    for row in data:
        if '#' in row[0]:
            y_vals = row[0].split("#")
            for val in y_vals:
                y.append(val)
                x.append(row[1])
        else:
            x.append(row[1])
            y.append(row[0])
    x = np.array(x)
    y = np.array(y)
    return x, y


class AtisDataset():
    """
    The Bank Marketing dataset comes from a phone-based marketing campaign from a Portuguese banking institution.
    The labels indicate whether an individual contacted in the campaign chose to subscribe to the product (bank term deposit) or not.

    See :file:`uq360/data/banking_data/README.md` for more details on the dataset and instructions on downloading/processing the data.
    References:
        .. [#] `[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
         <https://archive.ics.uci.edu/ml/machine-learning-databases/00222/>`_
    """

    def __init__(self, custom_preprocessing=default_preprocessing, dirpath=None):
        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data', 'atis_text_data')

        self._filepath = os.path.join(self._dirpath, 'atis_intents.csv')
        try:
            data = []
            with open(self._filepath, "r") as f:
                reader = csv.reader(f, delimiter=',', quotechar='|')
                for row in reader:
                    data.append(row)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please place the atis_intents.csv:")
            print("file, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '../data', 'atis_text_data'))))
            print("See :file:`uq360/data/atis_text_data/README.md` for more details on the dataset and instructions "
                  "on downloading/processing the data.")
            sys.exit(1)

        if custom_preprocessing:
            self._data = custom_preprocessing(data)

    def data(self):
        return self._data