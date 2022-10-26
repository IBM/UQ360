import unittest

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def create_train_test_prod_split(x, y, test_size=0.25):
    """
    returns x_train, y_train, x_test, y_test, x_prod, y_prod
    """
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


def split(x, y, bucket_1_indices, bucket_2_indices, split_ratio=0.3, test_size=0.25):
    """
    returns: x_train, y_train, x_test, y_test, x_prod, y_prod
    """
    train_test_samples = x.shape[0] * 0.5
    training_test_data_from_bucket_1 = np.random.choice(bucket_1_indices[0], int(train_test_samples * split_ratio),
                                                        replace=False)
    training_test_data_from_bucket_2 = np.random.choice(bucket_2_indices[0],
                                                        int(train_test_samples * (1 - split_ratio)), replace=False)

    prod_data_from_bucket_1 = np.setdiff1d(bucket_1_indices, training_test_data_from_bucket_1)
    prod_data_from_bucket_2 = np.setdiff1d(bucket_2_indices, training_test_data_from_bucket_2)

    training_test_data_indices = np.concatenate((training_test_data_from_bucket_1, training_test_data_from_bucket_2),
                                                axis=0)
    prod_indices = np.concatenate((prod_data_from_bucket_1, prod_data_from_bucket_2), axis=0)

    training_test_data = x[training_test_data_indices]
    training_test_label = y[training_test_data_indices]

    prod_test_data = x[prod_indices]
    prod_test_label = y[prod_indices]

    from sklearn.model_selection import train_test_split
    x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(training_test_data, training_test_label,
                                                                        test_size=test_size,
                                                                        random_state=42)

    print("Training data size:", x_train_new.shape)
    print("Test data size:", x_test_new.shape)
    print("Prod data size:", prod_test_data.shape)

    return x_train_new, y_train_new, x_test_new, y_test_new, prod_test_data, prod_test_label


class PlusOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


def base_test_case(cls):
    """Decorator for TestCases that are base classes not to be run"""

    def setUpClass(my_cls):
        if my_cls is cls:
            raise unittest.SkipTest("Skipping base class LatentScorerTester")
        super().setUpClass()

    cls.setUpClass = classmethod(setUpClass)
    return cls
