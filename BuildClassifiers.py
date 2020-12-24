from LogisticRegression import LogisticRegression
from NaiveBayesImp import NaiveBaseImp
from TxtDataProcess import TxtDataProcess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from time import time

# path to database as argument ../path/
print(f'|----------------------------------------------------------------------------------------------------------------|')
print(f'|----------------------------------------Load Database-----------------------------------------------------------|')
dataSet = TxtDataProcess(r"D:\University\Artificial_Intelligence/3130212\src\python\enron2\enron2\ham/")
dataSet.addDataFromTxtFiles()

print(f'|----------------------------------------------------------------------------------------------------------------|')
print(f'|---------------------------Split train data to train-dev-test data----------------------------------------------|')
dataSet.split_train_dev_test()

print(f'|----------------------------------------------------------------------------------------------------------------|')
print(f'|---------------------------------Train Naive Bayes Classifier---------------------------------------------------|')
clf1 = NaiveBaseImp()
clf1.train(dataSet._train_samples_data, dataSet._train_samples_category)

print(f'|----------------------------------------------------------------------------------------------------------------|')
print(f'|--------------------------Calculate Information Gaining for attributes------------------------------------------|')
clf1.calculateIG()

print(f'|----------------------------------------------------------------------------------------------------------------|')
print(f'|------------------------------Train LogisticRegression Classifier-----------------------------------------------|---AC-----ER-----F1-----RE-----PR---|')
clf = LogisticRegression()
clf.create_features([x for x in clf1.ig.keys()][:2000])
w = clf.train(dataSet._train_samples_data, dataSet._train_samples_category, dataSet)

print(f'|----------------------------------------------------------------------------------------------------------------|')
print(f'|-------------------------------Test LogisticRegression Classifier-----------------------------------------------|')
clf.test(dataSet._dev_samples_data, dataSet._dev_samples_category, w, True)

print(f'|----------------------------------------------------------------------------------------------------------------|')
print(f'|----------------------------------Test Naive Bayes Classifier---------------------------------------------------|')
clf1.test(dataSet._dev_samples_data, dataSet._dev_samples_category)
print(f'|----------------------------------------------------------------------------------------------------------------|')

#############################
# use sklearn lib
print()
print()
print(f'|---SKLEARN RESULTS---|')
features_train, features_test, labels_train, labels_test = train_test_split(dataSet._samplesData, dataSet._samplesCategory, test_size=0.2, random_state=4)

### text vectorization--go from strings to lists of numbers
vectorizer = TfidfVectorizer()
features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed  = vectorizer.transform(features_test)

### feature selection, because text is super high dimensional and 
    ### can be really computationally chewy as a result
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train_transformed, labels_train)
features_train_transformed = selector.transform(features_train_transformed).toarray()
features_test_transformed  = selector.transform(features_test_transformed).toarray()

from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
clf = GaussianNB()
clf.fit(features_train_transformed, labels_train)
pred = clf.predict(features_test_transformed)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print(accuracy)

reg = linear_model.LogisticRegression() 
reg.fit(features_train_transformed, labels_train)
y_pred = reg.predict(features_test_transformed)
accuracy = accuracy_score(y_pred, labels_test)
print(accuracy)