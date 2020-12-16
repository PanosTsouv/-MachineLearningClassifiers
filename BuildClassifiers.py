from TxtDataProcess import TxtDataProcess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

# path to database as argument ../path/
dataSet = TxtDataProcess(input("Enter DataSet's path: "))
dataSet.addDataFromTxtFiles()
dataSet.split_train_test()

#############################
# use sklearn lib
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
from time import time
clf = GaussianNB()
t0 = time()
clf.fit(features_train_transformed, labels_train)
print("training time:", round(time()-t0, 3), "s")
t1 = time()
pred = clf.predict(features_test_transformed)
print("pred time:", round(time()-t1, 3), "s")
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print(accuracy)