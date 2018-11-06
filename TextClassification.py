import os
import nltk
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

# print title and category
def print_plot(index):
    examples = news[news.article_id == index][['title','category']].values[0]
    if len(examples) > 0:
        print(examples[0])
        print('category:', examples[1])

# plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def prepareDatasets(input):
    # clean up texts
    # category - Article category (0 = Ratings downgrade, 1 = Sanctions, 2 = Growth into new markets, 3 = New product coverage, 4 = Others)
    # df = pd.read_csv("Test_data/test_v2.csv")
    df = pd.read_csv(input)

    # remove punctuation, digits, double spaces, stem and stop words
    stop = set(stopwords.words('english'))
    snow = SnowballStemmer('english')
    df['title'] = df.title.map(lambda x: x.lower())
    # df['title'] = df.title.map(lambda x: x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))))
    df['title'] = df.title.map(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df['title'] = df.title.map(lambda x: x.translate(str.maketrans('', '', string.digits)))
    df['title'] = df.title.map(lambda x: ' '.join(x.split()))
    df['title'] = df.title.map(lambda x: [snow.stem(word) for word in x.split() if word not in stop])

    temp = []

    for row in df['title']:
        seq = ''
        for word in row:
            seq = seq + ' ' + word

        temp.append(seq)

    df['title'] = temp
    X = df['title']

    return X

# clean up texts
# category - Article category (0 = Ratings downgrade, 1 = Sanctions, 2 = Growth into new markets, 3 = New product coverage, 4 = Others)
news = pd.read_csv("Test_data/train_v2.csv")

# remove punctuation, digits, double spaces, stem and stop words
stop = set(stopwords.words('english'))
snow = SnowballStemmer('english')
news['title'] = news.title.map(lambda x: x.lower())
# news['title'] = news.title.map(lambda x: x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))))
news['title'] = news.title.map(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
news['title'] = news.title.map(lambda x: x.translate(str.maketrans('', '', string.digits)))
news['title'] = news.title.map(lambda x: ' '.join(x.split()))
news['title'] = news.title.map(lambda x: [snow.stem(word) for word in x.split() if word not in stop])

temp = []
for row in news['title']:
    seq = ''
    for word in row:
        seq = seq + ' ' + word

    temp.append(seq)
news['title'] = temp

news['title']

# handle imblanced datasets
X = news['title']
y = news['category']
labels = ['Ratings downgrade','Sanctions','Growth into new markets','New product coverage','Others']
print('Training class distribution: {}'.format(Counter(y)))
# y.value_counts().plot(kind='bar')

# vectorize and resampling with SMOTE
tfidf_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
X_tfidf = tfidf_vec.fit_transform(X)
X_s_tfidf = tfidf_vec.transform(prepareDatasets("Test_data/test_v2.csv"))
print('Training: ', X_tfidf.shape)
print('Test: ', X_s_tfidf.shape)
ros = ADASYN(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_tfidf, y)
print(sorted(Counter(y_resampled).items()))

# split training data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)
print('Training class distribution: {}'.format(Counter(y_train)))

# Build Model
clf = LinearSVC().fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('accuracy: ', accuracy_score(y_pred, y_test))
print(classification_report_imbalanced(y_true=y_test, y_pred=y_pred))

# confusion matrix
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
plt.figure()
plot_confusion_matrix(conf_mat, classes=labels, normalize=False, title='Confusion matrix, without normalization')

# test
y_submit = clf.predict(X_s_tfidf)

# export to csv
df = pd.DataFrame(data=y_submit)
df.to_csv("Test_data/Submission.csv")


10850*3000
11000*3000
