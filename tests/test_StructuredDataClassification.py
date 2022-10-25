import os
import unittest
from unittest import TestCase

import requests
import zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tests.utils import create_train_test_prod_split
from uq360.algorithms.blackbox_metamodel.structured_data_classification import StructuredDataClassificationWrapper


class TestStructuredDataClassification(TestCase):

    def test_structured_data_pred(self):

        x, y = self.get_banking_data()
        # create a random train/test/prod split
        x_train, y_train, x_test, y_test, x_prod, y_prod = create_train_test_prod_split(x, y)

        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)

        acc_on_test = rf.score(x_test, y_test)
        acc_on_prod = rf.score(x_prod, y_prod)

        p = StructuredDataClassificationWrapper(base_model=rf)
        p.fit(x_train, y_train, x_test, y_test)

        # predict the model's accuracy on production/unlabeled data
        y_mean, y_pred, y_score = p.predict(x_prod)
        print("Accuracy on prod data", acc_on_prod * 100)
        print("Predictor's prediction", y_mean)

        delta = abs(y_mean - acc_on_prod * 100)
        self.assertTrue(delta <= 3)

    def get_banking_data(self):

        columns = {
            "ordered_categorical_columns": ["education"],
            "categorical_columns": ["job", "marital", "housing", "loan", "contact", "poutcome"],
            "numerical_columns": ["age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
                                  "cons.conf.idx", "euribor3m", "nr.employed"],
            "text_columns": [],
            "ignore_columns": ["default", "month", "day_of_week"],
            "targets": "y"}
        file = {"filename": "bank-additional-full.csv", "separator": ";", "extension": ".csv"}
        separator = file.get('separator', ',')

        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
        r = requests.get(url, allow_redirects=True)

        filename = os.path.join(os.getcwd(),'uq360/data/banking_data/bank-additional.zip')

        open(filename, 'wb').write(r.content)
        assert os.path.exists(filename)

        banking_dir = filename.rsplit("/", 1)[0]
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extract("bank-additional/bank-additional-full.csv", banking_dir)

        csv_file = os.path.join(banking_dir, "bank-additional", "bank-additional-full.csv")
        assert os.path.exists(csv_file)

        df = pd.read_csv(csv_file, sep=separator, na_values=["unknown"], engine='python')

        df = df[:5000]
        df = df.drop(columns.get('ignore_columns', []), axis=1)

        df = df.dropna(axis=0, how='any')

        # Get target value 'y'
        y_label = columns.get("targets", 'y')
        y = df[y_label].values
        df = df.drop([y_label], axis=1)  # Remove labels from feature set x'

        class_dict = {}
        for ind, label in enumerate(np.unique(y)):
            class_dict[ind] = str(label)

        # Encode labels as int
        le = LabelEncoder()
        y = le.fit_transform(y)

        metadata = {}
        # identify categorical vs numerical columns
        categorical_columns = columns.get("categorical_columns", [])
        ordered_categorical_columns = columns.get("ordered_categorical_columns", [])
        # use label encoder to convert string into numerical values
        ohe = OneHotEncoder(categories='auto', sparse=False)
        if categorical_columns or ordered_categorical_columns:  # if there are categorical variables, store details
            metadata['categorical_column_details'] = {}
        features = {}

        for column in df.columns.values:
            if column in ordered_categorical_columns:
                features[str(column)] = le.fit_transform(df[column].values.astype('str')).reshape(-1, 1)
                metadata['categorical_column_details'][column] = list(le.classes_)
            elif column in categorical_columns:
                features[str(column)] = ohe.fit_transform(df[column].values.astype('str').reshape(-1, 1))
            else:  # By default if not labeled as categorical, treat as numeric
                features[str(column)] = df[column].values.astype('float').reshape(-1, 1)
        x = np.concatenate([np.array(v, dtype=float) for k, v in features.items()], axis=1)

        return x, y


if __name__ == '__main__':
    unittest.main()