from random import seed
import random
from time import time

import numpy as np

class DataSet:
    def __init__(self, path) -> None:
        self._path = path
        self._samplesPerCategory = {}
        self._samplesData = []
        self._samplesCategory = []
        self._train_samples_data = []
        self._train_samples_category = []
        self._test_samples_data = []
        self._test_samples_category = []
        self._dev_samples_data = []
        self._dev_samples_category = []


    def dataSetInfo(self):
        for category, samples in self._samplesPerCategory.items():
            print(f'Category {category} has {samples} samples')
        print(f'All samples array category : {self._samplesCategory}')
        print(f'Train array category : {self._train_samples_category}')
        print(f'Dev array category : {self._dev_samples_category}')
        print(f'Test array category : {self._test_samples_category}')

    def split_train_dev_test(self, test_size  = 0, dev_size = 0.2, random_number = 5, percentage_of_sample = 0.8):
        test_size_hidden = 0
        if (test_size + dev_size) >= percentage_of_sample: test_size, dev_size = 0.1, 0.1

        if percentage_of_sample < 1:
            test_size_hidden = 10 - percentage_of_sample*10
            test_size_hidden = test_size_hidden / 10

        if test_size != 0: test_size_hidden = test_size

        if random_number != -1: np.random.seed(random_number)
        
        temp_array_samples = np.array(self._samplesData)
        temp_array_samples_cat = np.array(self._samplesCategory)

        for label in self._samplesPerCategory.keys():
            array_per_label = np.where(np.array(self._samplesCategory) == label)[0]
            size = array_per_label.size

            sampling_dev = np.random.choice(array_per_label, replace=False, size = (int(size*dev_size) + int(size*test_size_hidden)))
            array_per_label = array_per_label[~np.isin(array_per_label,sampling_dev)]

            self._dev_samples_category.extend(temp_array_samples_cat[sampling_dev[0:int(size*dev_size)]])
            self._dev_samples_data.extend(temp_array_samples[sampling_dev[0:int(size*dev_size)]])
            if test_size != 0:
                self._test_samples_category.extend(temp_array_samples_cat[sampling_dev[int(size*dev_size):]])
                self._test_samples_data.extend(temp_array_samples[sampling_dev[int(size*dev_size):]])
            self._train_samples_category.extend(temp_array_samples_cat[array_per_label])
            self._train_samples_data.extend(temp_array_samples[array_per_label])