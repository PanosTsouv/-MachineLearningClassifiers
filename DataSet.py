from random import seed
import random
from time import time

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


    def dataSetInfo(self):
        for category, samples in self._samplesPerCategory.items():
            print(f'Category {category} has {samples} samples')
        print(f'All samples array category : {self._samplesCategory}')
        print(f'Train array category : {self._train_samples_category}')
        print(f'Test array category : {self._test_samples_category}')


    def split_train_test(self, test_size  = 0.2, random_number = 1):

        t0 = time()
        # number of test samples per category
        test_samples_per_category = self._number_of_test_samples_per_cat(test_size)
        # dir contain all position's samples per category
        positions_of_samples_per_category = self._positions_of_samples_per_cat()
        # list contain all position's samples which add in test_set
        all_selections = self._pick_random_samples(random_number, positions_of_samples_per_category, test_samples_per_category)
        self._train_samples_category = [sample_cat for sample_cat in self._samplesCategory]
        self._train_samples_data = [sample_data for sample_data in self._samplesData]
        for i_sel in all_selections:
            self._test_samples_data.append(self._samplesData[i_sel])
            self._test_samples_category.append(self._samplesCategory[i_sel])
            del self._train_samples_category[i_sel]
            del self._train_samples_data[i_sel]
        print("My split time:", round(time()-t0, 3), "s")


    def _pick_random_samples(self, random_number, pos_sequence_per_category, test_samples_per_category):
        seed(random_number)
        all_selections_pos = []
        for i_category in self._samplesPerCategory.keys():
            sequence = pos_sequence_per_category[i_category]
            selection = random.sample(sequence, test_samples_per_category[i_category])
            all_selections_pos += selection
        all_selections_pos = sorted(all_selections_pos, reverse=True)
        return all_selections_pos


    def _number_of_test_samples_per_cat(self, test_size):
        test_samples_per_category = {}
        for i_category in self._samplesPerCategory.keys():
            test_samples_per_category[i_category] = int(self._samplesPerCategory[i_category]*test_size)
        return test_samples_per_category


    def _positions_of_samples_per_cat(self):
        temp = {}
        for i_category in self._samplesPerCategory.keys():
            temp[i_category] = [i_data for i_data in range(len(self._samplesData)) if self._samplesCategory[i_data] == i_category]
        return temp