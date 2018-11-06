import pandas as pd

train = "train_v2.csv"

dat = pd.read_csv(train,encoding='latin-1')

# Cleaning
dat['title'] = dat['title'].str.replace(r'[^\w\s]+', '')
dat['publisher'] = dat['publisher'].str.replace(r'[^\w\s]+', '')

## Remove digits

dat['title'] = dat['title'].str.lower()
dat['publisher'] = dat['publisher'].str.lower()

dat['title'] = dat['title'].replace('\s+', ' ', regex=True)
dat['publisher'] = dat['publisher'].replace('\s+', ' ', regex=True)

dat.drop(['article_id','url','hostname', 'timestamp'], inplace=True, axis=1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    dat['title'], 
    dat['category'], 
    random_state = 1
)

print("Training dataset: ", X_train.shape[0])
print("Test dataset: ", X_test.shape[0])

# Vectorizer
from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer(stop_words = 'english')
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)


# Topic Modeling


# Train - classifier 1
from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

# Predict
predictions = naive_bayes.predict(testing_data)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print("Accuracy score: ", accuracy_score(y_test, predictions))
print("Recall score: ", recall_score(y_test, predictions, average = 'weighted'))
print("Precision score: ", precision_score(y_test, predictions, average = 'weighted'))
print("F1 score: ", f1_score(y_test, predictions, average = 'weighted'))

# Train - classifier 2
from sklearn import svm

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(training_data, y_train)

predictions = clf.predict(testing_data)

print("Accuracy score: ", accuracy_score(y_test, predictions))
print("Recall score: ", recall_score(y_test, predictions, average = 'weighted'))
print("Precision score: ", precision_score(y_test, predictions, average = 'weighted'))
print("F1 score: ", f1_score(y_test, predictions, average = 'weighted'))




# Import Test
test = "/Users/valerieho/Documents/09 NUS/Knowledge Discovery/Group Project/test_v2.csv"
dat_test = pd.read_csv(test,encoding='latin-1')

# Pre-Process Test

dat_test['title'] = dat_test['title'].str.replace(r'[^\w\s]+', '')
dat_test['publisher'] = dat_test['publisher'].str.replace(r'[^\w\s]+', '')

## Remove digits

dat_test['title'] = dat_test['title'].str.lower()
dat_test['publisher'] = dat_test['publisher'].str.lower()

dat_test['title'] = dat_test['title'].replace('\s+', ' ', regex=True)
dat_test['publisher'] = dat_test['publisher'].replace('\s+', ' ', regex=True)

dat_test.drop(['article_id','url','hostname', 'timestamp'], inplace=True, axis=1)

X_test = dat_test['title']

testing_data = count_vector.transform(X_test)

# Predict
predictions = naive_bayes.predict(testing_data)

predictions = pd.DataFrame(predictions)
predictions.reset_index(level=0, inplace=True)
predictions.columns = ['article_id','category']

predictions['article_id'] = predictions['article_id'] + 1

predictions.to_csv("predictions.csv", index=False)

