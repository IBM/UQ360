import os
import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from uq360.algorithms.blackbox_metamodel.structured_data_predictor import StructuredDataPredictorWrapper


class TestBlackBoxMetamodelClassification(TestCase):
    def _generate_mock_data(self, n_samples, n_classes, n_features):
        from sklearn.datasets import make_classification
        return make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                   n_informative=n_features, n_redundant=0, random_state=42, class_sep=10)

    def create_train_test_prod_split(self, x, y, test_size=0.25):
        """
        returns x_train, y_train, x_test, y_test, x_prod, y_prod
        """
        from sklearn.model_selection import StratifiedKFold, train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.25,
                                                            random_state=42)

        x_test, x_prod, y_test, y_prod = train_test_split(x_test, y_test,
                                                          test_size=0.25,
                                                          random_state=42)

        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_prod.shape, y_prod.shape)

        print("Training data size:", x_train.shape)
        print("Test data size:", x_test.shape)
        print("Prod data size:", x_prod.shape)

        return x_train, y_train, x_test, y_test, x_prod, y_prod

    def test_structured_data_pred(self):

        x, y = self.get_banking_data()
        # create a random train/test/prod split
        x_train, y_train, x_test, y_test, x_prod, y_prod = self.create_train_test_prod_split(x, y)

        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)

        acc_on_test = rf.score(x_test, y_test)
        acc_on_prod = rf.score(x_prod, y_prod)

        p = StructuredDataPredictorWrapper(base_model=rf)
        p.fit(x_train, y_train, x_test, y_test)

        # predict the model's accuracy on production/unlabeled data
        prediction = p.predict(x_prod)
        print("Accuracy on prod data", acc_on_prod * 100)
        print("Predictor's prediction", prediction)

        delta = abs(prediction["predicted_accuracy"] - acc_on_prod * 100)
        self.assertTrue(delta <= 0.5)

    def get_banking_data(self):
        local_file = os.path.abspath(
            os.path.join(os.getcwd(), "..", "data", "structured_data", "banking", "bank-additional-full.csv.gz"))
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

        df = pd.read_csv(local_file, sep=separator, na_values=["unknown"], engine='python', compression="gzip")

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