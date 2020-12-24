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

    """
        Fit the classifier with train data, parse them and calculate
        the weights of voc attributes for every sample.

        Parameters
        ----------
        train_data  : list of strings
            Every position is a sample - n size
        train_labels  : list of strings
            Every position is the label of sample - n size
        dataSet  : Dataset object
            Contains dev data for evaluate at every step(optional)[default : None]
        h : float
            Learning rate
        iterations : int
            Number of times algorith parse the training data
        l : float > 0
            Use for regularization
        
        Returns
        -------
        w : array of float
            Calculated weights for attributes
    """
    def train(self, train_data, train_labels, dataSet = None, h = 0.01, iterations  = 2000, percentage = 1000, l=0.1):
        w, certainty, self.b = np.zeros(len(self.voc)), np.zeros(len(train_data)), 0
        self.likelihood_ = []
        count = 0
        #convert list to np array X = (m_samples,n attributes) Y = (m_labels,)
        X, Y = self._samples_as_np_array(train_data, train_labels)

        progress = tqdm(total = iterations, desc=f'|-Sension {count}', ncols=138, ascii='->', colour='RED')
        while(count < iterations):
            #test every interaction (optional)
            if count % 100 == 0 and dataSet != None:
                self.test(dataSet._dev_samples_data[0:], dataSet._dev_samples_category[0:], w, False)
            self._update_progress_bar(progress, 1)
            #call sigmoid function
            certainty = self._find_certainty((X @ w) + self.b)
            #calculate gradient
            tmp = Y - certainty
            w = w + h * np.dot(X.T , tmp) - 2 * h * l * w
            self.b = self.b + h * np.sum( tmp )
            #calculate likelihood
            likelihood = np.dot(Y, np.log(certainty + 1e-5)) + np.dot(1-Y, np.log(1 + 1e-5  - certainty))
            likelihood -= self._regularization(w, l)
            self.likelihood_.append(likelihood)
            count += 1
        progress.close()
        return w


    """
        Test the given attribute and predict the label for it

        Parameters
        ----------
        test_data  : list of strings
            Every position is a sample - n size
        test_labels  : list of strings
            Every position is the label of sample - n size
        w  : array of float
            Calculated weights for attributes from train stage
        progress_bar_flag : boolean
            Show the progress of test and results[Default : True]
    """
    def test(self, test_data, test_labels, w, progress_bar_flag = True):
        if len(test_data) == 0: return
        self.algAnswers = []
        X, Y = self._samples_as_np_array(test_data, test_labels)
        progress = range(X.shape[0])
        if progress_bar_flag: progress = tqdm(total = X.shape[0], desc=f'|-Classifier', ncols=151, ascii='->', colour='green')
        for idx in range(X.shape[0]):
            if self._find_certainty(X[idx].dot(w)) > 0.5:
                self.algAnswers.append(1)
            else:
                self.algAnswers.append(0)
            self._calculateResults(test_labels)
            if progress_bar_flag:
                progress.bar_format = '{l_bar}{bar}{r_bar}' + '- '  + str("{:.2f}".format(self.acc)) + ' - ' + str("{:.2f}".format(self.error)) + ' - '  
                progress.bar_format += str("{:.2f}".format(self.F1)) + ' - '  + str("{:.2f}".format(self.recall)) + ' - '  + str("{:.2f}".format(self.precision)) + ' -|'
                progress.update()
        if progress_bar_flag: progress.close()

    """
        Convert a list to np array (m,n) for data (m,)

        Parameters
        ----------
        data  : list of strings
            Every position is a sample - n size
        labels  : list of strings
            Every position is the label of sample - n size

        Returns
        -------
        X : array - shape(m,n)
            Every row is a sample and every column is an attribute
    """
    def _samples_as_np_array(self, data, labels):
        Y = np.array(labels[0:])
        X = np.empty([len(data[0:]),len(self.voc)])
        for sample_index in range(len(data[0:])):
            sample_as_attr = np.zeros(len(self.voc))
            words = set(data[sample_index].split())
            for word in words:
                if self.voc.get(word,0) != 0:
                    sample_as_attr[self.voc[word]] = 1
            X[sample_index] = sample_as_attr
        return X,Y

    def _update_progress_bar(self, progress, x):
        if len(self.likelihood_) > 2:
            progress.ncols = 152 + len(str("{:.2f}".format(self.likelihood_[-2]-self.likelihood_[-1]))) + len(str("{:.2f}".format(self.likelihood_[-1])))
            progress.bar_format = '{l_bar}{bar}{r_bar}' + '- '  + str("{:.2f}".format(self.acc)) + ' - ' + str("{:.2f}".format(self.error)) + ' - '
            progress.bar_format += str("{:.2f}".format(self.F1)) + ' - '  + str("{:.2f}".format(self.recall)) + ' - '  + str("{:.2f}".format(self.precision)) + ' -|'
            progress.bar_format += str("{:.2f}".format(self.likelihood_[-2]-self.likelihood_[-1])) + '|' + str("{:.2f}".format(self.likelihood_[-1]))
        progress.update(x)

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

    def _find_certainty(self, x):
        pCP_X = 1 / (1 + np.exp(-x))
        return pCP_X

    def _regularization(self, w, l):
        return l * sum(w**2)#np.dot(w, w)

    def _info(self, train_labels, sample_index, certainty, lw, s):
        print(F'Label of sample is {train_labels[sample_index]}')
        if train_labels[sample_index]:
            print(f'certainty of sample {sample_index} is {certainty[sample_index]}')
            print(f'Log of certainty of {sample_index} is {np.log(certainty[sample_index]+0.00001)}')
        else:
            print(f'certainty of sample {sample_index} is {1-certainty[sample_index]}')
            print(f'Log of certainty of {sample_index} is {np.log(1.00001 - certainty[sample_index])}')
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
