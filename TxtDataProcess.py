import re
from time import sleep
from DataSet import DataSet
import os
import sys
from tqdm import tqdm

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
        self.stop_words = self._readTxTFile(r'StopWords.txt')

    def _readTxTFile(self,filename):
        f = open(self._path + filename, "r")
        words = f.read().split()
        f.close()
        return words
    
    def addDataFromTxtFiles(self, number_of_samples = {1:-1, 0:-1}):
        directory = os.listdir(self._path)
        progress = tqdm(range(len(directory)), desc=f'|-Load Data', ncols=130, ascii='->', colour='yellow')
        for filename_index in progress:
            progress.ncols = 149 if directory[filename_index].find('.ham') != -1 else 151
            progress.update()
            if directory[filename_index].endswith(".txt"):
                progress.bar_format = '{l_bar}{bar}{r_bar}' + str('---' + directory[filename_index])
                progress.update()
                words = self._readTxTFile(directory[filename_index])
                if self._check_number_of_samples(number_of_samples): break
                if directory[filename_index].find('.ham') != -1:
                    if not self._create_samples_data(number_of_samples[1], 1, words): continue
                else:
                    if not self._create_samples_data(number_of_samples[0], 0, words): continue
            else:
                continue
            if filename_index == len(directory)-1:
                progress.ncols = 130
                progress.update()

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
            if len(word) > 1 and not re.search(r'\d', word) and not word in self.stop_words:
                tempStr += (' ' + word)
        return tempStr

