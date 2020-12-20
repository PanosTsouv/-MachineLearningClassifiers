from time import sleep, time
import numpy as np


class NaiveBaseImp:

    def __init__(self, label_cls = [0, 1], attribute_cls = [0, 1]) -> None:
        self.true_pos = 0;
        self.false_pos = 0;
        self.true_neg = 0;
        self.false_neg = 0;
        self.label_cls = label_cls
        self.attribute_cls = attribute_cls
        self.num_samples_per_label = {}
        self.num_train_samples = 0
        self.train_labels = []
        self.X = []
        self.features = {}
        self.ig = {}
        self.rightAnswers = []
        self._acc = 0.0



    def train(self, train_attributes_samples, train_labels):
        for label in self.label_cls:
            for i in range(len(train_labels)):
                if label == train_labels[i]:
                    self.num_samples_per_label[label] = self.num_samples_per_label.get(label, 0) + 1
        
        print(self.num_samples_per_label)
        self.num_train_samples = len(train_attributes_samples)
        self.X = train_attributes_samples
        self.train_labels = train_labels
        self._create_features()
        
    def test(self, test_attributes_samples, test_labels):
        t0 = time()
        for sample in test_attributes_samples:
            self.rightAnswers.append(self._classify(sample))
        print(len(self.features))
        self._calculateResults(test_labels)
        print(self.acc)
        print("Test time", round(time()-t0, 3), "s")

    def _classify(self, sample, number_of_attributes = 1000):
        pC = {}
        pXC = {}
        pCX = {0: 1, 1: 1}
        for label in self.label_cls:
            pC[label] = self.num_samples_per_label[label]/self.num_train_samples
        count = 0
        for word in self.ig:
            if count == number_of_attributes: break
            for label in self.label_cls:
                if word in sample:
                    pXC[label] = self.features[word].get(label,0)/self.num_samples_per_label[label]
                else:
                    pXC[label] = (self.num_samples_per_label[label]-self.features[word].get(label,0))/self.num_samples_per_label[label]
            for label in self.label_cls:
                log_prob = np.log2(pXC[label])
                pCX[label] += log_prob
            count += 1
        max = -100000000000000000000000
        maxLabel = ''
        for label in self.label_cls:
            pCX[label] += np.log2(pC[label])
            if pCX[label] > max:
                max = pCX[label]
                maxLabel = label
        return maxLabel

    def _create_features(self):
        t0 = time()
        for index_sample in range(self.num_train_samples):
            words = set(self.X[index_sample].split())
            for word in words:
                if self.features.get(word,0) == 0:
                    number_find_per_category = {}
                    self.features[word] = number_find_per_category
                if self.features[word].get(self.train_labels[index_sample], 0) == 0:
                    self.features[word][self.train_labels[index_sample]] = 0
                self.features[word][self.train_labels[index_sample]] = self.features[word].get(self.train_labels[index_sample], 0) + 1
        print("create_feature time:", round(time()-t0, 3), "s")

    def calculateIG(self):
        self._laplace_smooth(2)
        pX0, pX1, hC, self.num_train_samples = 0.0, 0.0, 0.0, 0.0
        for category in self.label_cls:
            self.num_train_samples += self.num_samples_per_label[category]
        for category in self.label_cls:
            hC -= self.num_samples_per_label[category]/self.num_train_samples*np.log2(self.num_samples_per_label[category]/self.num_train_samples)
        for feature in self.features.keys():
            pCX1, pCX0 = {}, {}
            term_number, hCx0, hCx1 = 0.0, 0.0, 0.0
            for category in self.label_cls:
                pCX1[category], pCX0[category] = 0, 0
                term_number += self.features.get(feature,0).get(category, 0)
            pX1 = term_number/self.num_train_samples
            pX0 = 1.0 - pX1
            for category in self.label_cls:
                pCX1[category] = self.features.get(feature,0).get(category, 0)/term_number
                pCX0[category] = (self.num_samples_per_label[category]-self.features.get(feature).get(category, 0))/(self.num_train_samples-term_number)
            for category in self.label_cls:
                hCx0 -= pCX0[category]*np.log2(pCX0[category])
                hCx1 -= pCX1[category]*np.log2(pCX1[category])
            self.ig[feature] = hC - (pX0*hCx0)-(pX1*hCx1)
        self.ig = dict(sorted(self.ig.items(), key=lambda item: item[1], reverse=True))

    def _laplace_smooth(self, attributes_class = 2):
        for category in self.label_cls:
            self.num_samples_per_label[category] = self.num_samples_per_label[category] + attributes_class
        print(self.num_samples_per_label)
        for feature in self.features:
            for category in self.label_cls:
                self.features.get(feature)[category] = self.features.get(feature).get(category,0) + 1

    def _calculateResults(self, test_labels):
        for i in range(len(self.rightAnswers)):
            if self.rightAnswers[i] == test_labels[i]:
                self.true_pos += 1.0
    @property
    def acc(self):
        print(self.true_pos)
        print(len(self.rightAnswers))
        self._acc = self.true_pos / len(self.rightAnswers)
        return self._acc