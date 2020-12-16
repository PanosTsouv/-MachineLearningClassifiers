from DataSet import DataSet
import os
import sys
"""
this class read the database and update the DataSet object
so the methods of this class should be different in how
you can access your database's files
here we read data from many txt files which have
a specific word in their names so we can know 
about their category
"""
class TxtDataProcess(DataSet):
    def __init__(self, path) -> None:
        super().__init__(path)

    def _readTxTFile(self,filename):
        f = open(self._path + filename, "r")
        words = f.read().split()
        f.close()
        return words
    
    def addDataFromTxtFiles(self, number_of_ham_samples = 200, number_of_spam_samples = 200, number_of_random = 200):
        for filename in os.listdir(self._path):
            tempStr = ''
            if filename.endswith(".txt"):
                words = self._readTxTFile(filename)
                for word in words:
                    if len(word) > 1:
                        tempStr += (' ' + word)
                if self._samplesPerCategory.get(1, 0) == number_of_ham_samples and self._samplesPerCategory.get(0, 0) == number_of_spam_samples: break
                if filename.find('kaminski') != -1:
                    if self._samplesPerCategory.get(1, 0) == number_of_ham_samples:
                        continue
                    self._samplesData.append(tempStr)
                    self._samplesCategory.append(1)
                    self._samplesPerCategory[1] = self._samplesPerCategory.get(1, 0) + 1
                elif filename.find('GP') != -1 or filename.find('farmer') != -1:
                    if self._samplesPerCategory.get(2, 0) == number_of_random:
                        continue
                    self._samplesData.append(tempStr)
                    self._samplesCategory.append(2)
                    self._samplesPerCategory[2] = self._samplesPerCategory.get(2, 0) + 1
                elif filename.find('SA_and_HP') != -1:
                    if self._samplesPerCategory.get(0, 0) == number_of_spam_samples:
                        continue
                    self._samplesData.append(tempStr)
                    self._samplesCategory.append(0)
                    self._samplesPerCategory[0] = self._samplesPerCategory.get(0, 0) + 1
            else:
                continue
