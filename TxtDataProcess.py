from DataSet import DataSet
import os
import sys

class TxtDataProcess(DataSet):
    """
    this class read the database and update the DataSet object
    so the methods of this class should be different in how
    you can access your database's files \n
    we read data from many txt files which have
    a specific word in their names so we can know 
    about their category
    """
    def __init__(self, path) -> None:
        super().__init__(path)

    def _readTxTFile(self,filename):
        f = open(self._path + filename, "r")
        words = f.read().split()
        f.close()
        return words
    
    def addDataFromTxtFiles(self, number_of_samples = {1:5, 0:5, 2:5}):
        for filename in os.listdir(self._path):
            if filename.endswith(".txt"):
                words = self._readTxTFile(filename)
                if self._check_number_of_samples(number_of_samples): break
                if filename.find('kaminski') != -1:
                    if not self._create_samples_data(number_of_samples[1], 1, words): continue
                elif filename.find('GP') != -1 or filename.find('farmer') != -1:
                    if not self._create_samples_data(number_of_samples[2], 2, words): continue
                elif filename.find('SA_and_HP') != -1:
                    if not self._create_samples_data(number_of_samples[0], 0, words): continue
            else:
                continue

    def _create_samples_data(self, number_of_ham_samples, category, words):
        if self._samplesPerCategory.get(category, 0) == number_of_ham_samples:
            return False
        self._samplesData.append(self._parse_words(words))
        self._samplesCategory.append(category)
        self._samplesPerCategory[category] = self._samplesPerCategory.get(category, 0) + 1
        return True

    def _check_number_of_samples(self, number_of_samples):
        if (len(self._samplesPerCategory)) != len(number_of_samples):
            return False
        for category in self._samplesPerCategory.keys():
            if self._samplesPerCategory.get(category, 0) == 0: return False
            if self._samplesPerCategory.get(category, 0) != number_of_samples[category]:
                return False
        return True

    def _parse_words(self, words):
        tempStr = ''
        for word in words:
            if len(word) > 1:
                tempStr += (' ' + word)
        return tempStr

