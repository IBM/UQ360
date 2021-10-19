import os
import zipfile
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def default_preprocessing(df):
    """
    1. Drop some unnecessary columns and na values.
    2. Break off label column from dataframe.
    3. Encode categorical columns
    4. Return x and y as numpy arrays
    """

    # Data clean up
    df = df.drop(["default", "month", "day"], axis=1)
    df = df.dropna(axis=0, how='any')

    # Get target value 'y'
    y = df["y"].values
    df = df.drop(["y"], axis=1)  # Remove labels from feature set x'

    class_dict = {}
    for ind, label in enumerate(np.unique(y)):
        class_dict[ind] = str(label)

    # Encode labels as int
    le = LabelEncoder()
    y = le.fit_transform(y)

    features = list(df.columns.values)
    print('Features after dropping a few: ', features)

    # use label encoder to convert string into numerical values
    ohe = OneHotEncoder(categories='auto', sparse=False)
    features = {}

    for column in df.columns.values:
        # Label encoder for ordered categorical column
        if column == "education":
            features[str(column)] = le.fit_transform(df[column].values.astype('str')).reshape(-1, 1)
        # One hot encode categorical columns
        elif column in ["job", "marital", "housing", "loan", "contact", "poutcome"]:
            features[str(column)] = ohe.fit_transform(df[column].values.astype('str').reshape(-1, 1))
        else:  # By default if not labeled as categorical, treat as numeric
            features[str(column)] = df[column].values.astype('float').reshape(-1, 1)

    x = np.concatenate([np.array(v, dtype=float) for k, v in features.items()], axis=1)
    return x, y


class BankMarketingDataset():
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
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data', 'banking_data')

        self._filepath = os.path.join(self._dirpath, 'bank.zip')
        try:
            # read the dataset using the compression zip
            with zipfile.ZipFile(self._filepath) as z:
                # open the csv file in the dataset
                with z.open("bank-full.csv") as f:
                    # read the dataset
                    df = pd.read_csv(f, sep=';', na_values=["unknown"], engine='python')
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please place the bank.zip:")
            print("file, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '../data', 'banking_data'))))
            print("See :file:`uq360/data/banking_data/README.md` for more details on the dataset and instructions "
                  "on downloading/processing the data.")
            sys.exit(1)

        if custom_preprocessing:
            self._data = custom_preprocessing(df)

    def data(self):
        return self._data