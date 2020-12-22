from time import sleep, time
import numpy as np
from numpy import random
from tqdm import tqdm

class LogisticRegression:

    def __init__(self) -> None:
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0

    def train(self, train_data, train_labels, dataSet, h = 0.01, iterations  = 10, maxs = 100000, s = 0, l=0.1):
        w = np.zeros(len(self.voc)+1)
        likelihood = np.zeros(len(train_data))
        count = 0
        lw = 0
        while(count < iterations and s < maxs):
            progress = tqdm(range(len(train_data)), desc=f'|-Sension {count}', ncols=138, ascii='->', colour='RED')


            for sample_index in progress:
                sample_as_attr = np.zeros(len(self.voc)+1)
                sample_as_attr[-1] = 1
                words = set(train_data[sample_index].split())
                for word in words:
                    if self.voc.get(word,0) != 0:
                        sample_as_attr[self.voc[word]] = 1
                likelihood[sample_index] = self._find_likelihood(sample_as_attr, w)
                lw = train_labels[sample_index]*(np.log(likelihood[sample_index])) + (1 - train_labels[sample_index])*(np.log(1.00001 - likelihood[sample_index]))
                lw -= self._regularization(w,l)
                s -= lw
                for weight_index in range(len(w)):
                    w[weight_index] = w[weight_index] + h * (train_labels[sample_index] - likelihood[sample_index]) * (sample_as_attr[weight_index])


                if sample_index % 2300 == 0:
                    self.test(dataSet._dev_samples_data, dataSet._dev_samples_category, w, False)
                    progress.ncols = 151
                    progress.bar_format = '{l_bar}{bar}{r_bar}' + '- '  + str("{:.2f}".format(self.acc)) + ' - ' + str("{:.2f}".format(self.error)) + ' - '  + str("{:.2f}".format(self.F1)) + ' - '  + str("{:.2f}".format(self.recall)) + ' - '  + str("{:.2f}".format(self.precision)) + ' -|'
                    progress.update()
            count += 1
        return w

    def test(self, test_data, test_labels, w, progress_bar_flag = True):
        if len(test_data) == 0: return
        self.algAnswers = []
        progress = range(len(test_data))
        if progress_bar_flag:
            progress = tqdm(range(len(test_data)), desc=f'|-Classifier', ncols=151, ascii='->', colour='green')
        for sample_index in progress:
            sample_as_attr = np.zeros(len(self.voc)+1)
            sample_as_attr[-1] = 1
            words = set(test_data[sample_index].split())
            for word in words:
                if self.voc.get(word,0) != 0:
                    sample_as_attr[self.voc[word]] = 1
            if self._find_likelihood(sample_as_attr, w) > 0.5:
                self.algAnswers.append(1)
            else:
                self.algAnswers.append(0)
            self._calculateResults(test_labels)
            if progress_bar_flag:
                progress.bar_format = '{l_bar}{bar}{r_bar}' + '- '  + str("{:.2f}".format(self.acc)) + ' - ' + str("{:.2f}".format(self.error)) + ' - '  + str("{:.2f}".format(self.F1)) + ' - '  + str("{:.2f}".format(self.recall)) + ' - '  + str("{:.2f}".format(self.precision)) + ' -|'
                progress.update()

    def create_features(self, train_data):
        self.voc = {k: v for v, k in enumerate(train_data)}
        # print(f'length = {len(self.voc)}')
        # sleep(10)
        # t0 = time()
        # temp = []
        # for y in train_data:
        #     temp.extend(y.split())
        # self.voc = {k: v for v, k in enumerate(set(temp))}
        # print("Np array create attributes", round(time()-t0, 3), "s")

    def _find_likelihood(self, sample_as_attr, w):
        pCP_X = 1 / (1 + np.exp(-np.dot(w, sample_as_attr)))
        return pCP_X

    def _regularization(self, w, l):
        return l * np.dot(w, w)

    def _info(self, train_labels, sample_index, likelihood, lw, s):
        print(F'Label of sample is {train_labels[sample_index]}')
        if train_labels[sample_index]:
            print(f'Likelihood of sample {sample_index} is {likelihood[sample_index]}')
            print(f'Log of likelihood of {sample_index} is {np.log(likelihood[sample_index]+0.00001)}')
        else:
            print(f'Likelihood of sample {sample_index} is {1-likelihood[sample_index]}')
            print(f'Log of likelihood of {sample_index} is {np.log(1.00001 - likelihood[sample_index])}')
        print(f'Lw after regularization is : {lw}')
        print(f'Total s is : {s}')
        print()

    def _calculateResults(self, test_labels):
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        for i in range(len(self.algAnswers)):
            if test_labels[i] == 1 and self.algAnswers[i] == test_labels[i]:
                self.true_pos += 1.0
            elif test_labels[i] == 0 and self.algAnswers[i] == test_labels[i]:
                self.true_pos += 1.0
            elif test_labels[i] == 0 and self.algAnswers[i] == 1:
                self.false_pos += 1.0
            elif test_labels[i] == 1 and self.algAnswers[i] == 0:
                self.false_neg += 1.0

    @property
    def acc(self):
        if (self.true_pos + self.true_neg + self.false_pos + self.false_neg) == 0: return 0
        self._acc = (self.true_pos + self.true_neg) / (self.true_pos + self.true_neg + self.false_pos + self.false_neg)
        return self._acc

    @property
    def error(self):
        if (self.true_pos + self.true_neg + self.false_pos + self.false_neg) == 0: return 0
        self._error = (self.false_pos + self.false_neg) / (self.true_pos + self.true_neg + self.false_pos + self.false_neg)
        return self._error

    @property
    def F1(self):
        if (2 * self.true_pos + self.false_pos + self.false_neg) == 0: return 0
        self._F1 = 2 * self.true_pos / (2 * self.true_pos + self.false_pos + self.false_neg)
        return self._F1

    @property
    def recall(self):
        if (self.true_pos + self.false_neg) == 0: return 0
        self._recall = self.true_pos / (self.true_pos + self.false_neg)
        return self._recall

    @property
    def precision(self):
        if (self.true_pos + self.false_pos) == 0: return 0
        self._precision = self.true_pos / (self.true_pos + self.false_pos)
        return self._precision
