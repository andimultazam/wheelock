import pandas as pd
# import urllib
# import googlesearch # https://www.geeksforgeeks.org/performing-google-search-using-python-code/
# from urllib.error import URLError, HTTPError
# from urllib.request import Request, urlopen
# from http.client import IncompleteRead


# f1 = open("test_temp.txt",'w', encoding='utf-8')
train_file = pd.read_csv("train_v2.csv");
train_category = train_file.loc[:,'category'];

# tokenization (filtering stopwords included)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.pipeline import Pipeline

## Pre-process data
# Process title column
title_col = train_file.loc[:,'title'];
tcount_vect = CountVectorizer()
title_train_counts = tcount_vect.fit_transform(title_col.values.astype('U'))
print (title_train_counts.shape)
ttfidf_transformer = TfidfTransformer()
title_train_tfidf = ttfidf_transformer.fit_transform(title_train_counts)
print (title_train_tfidf.shape)


# Process publisher column
publisher_col = train_file.loc[:,'publisher'];
pcount_vect = CountVectorizer()
publisher_train_counts = pcount_vect.fit_transform(publisher_col.values.astype('U'))
print (publisher_train_counts.shape)
ptfidf_transformer = TfidfTransformer()
publisher_train_tfidf = ptfidf_transformer.fit_transform(publisher_train_counts)
print (publisher_train_tfidf.shape)

final_train_data = np.column_stack((title_train_tfidf.toarray(),publisher_train_tfidf.toarray()))

# clf = LinearSVC().fit(final_train_data, train_category)
clf = SGDClassifier().fit(final_train_data, train_category)

predicted = clf.predict(final_train_data)
print (list(train_category))
print (np.mean(predicted == train_category))

categories = {0:"Ratings downgrade", 1:"Sanctions", 2:"Growth into new markets", 3:"New product coverage", 4:"Others"};
# f.write("title\tpublisher\ttest_cat\tpred_cat\n")
# for i in range(len(train_file)):
# 	f.write(str(train_file.loc[i,'title']) + "\t" + str(train_file.loc[i,'publisher']) + "\t"+str(categories.get(int(train_file.loc[i,'category']))) +"\t" + str(categories.get(int(predicted[i]))) + "\n")

test_file = pd.read_csv('test_v2.csv');
title_col = test_file.loc[:,'title'];
title_train_counts = tcount_vect.transform(title_col.values.astype('U'))
# # print (title_train_counts.shape)
title_train_tfidf = ttfidf_transformer.transform(title_train_counts)
# print (title_train_tfidf.shape)


# Process publisher column
publisher_col = test_file.loc[:,'publisher'];
publisher_train_counts = pcount_vect.transform(publisher_col.values.astype('U'))
# # print (publisher_train_counts.shape)
publisher_train_tfidf = ptfidf_transformer.transform(publisher_train_counts)
# print (publisher_train_tfidf.shape)

final_train_data = np.column_stack((title_train_tfidf.toarray(),publisher_train_tfidf.toarray()))
predicted = clf.predict(final_train_data)
# f1.write("title\tpublisher\tpred_cat\n")

# for i in range(len(test_file)):
# 	f1.write(str(test_file.loc[i,'title']) + "\t" + str(test_file.loc[i,'publisher']) + "\t"+ str(categories.get(int(predicted[i]))) + "\n")
f = open("submission_v2.csv",'w', encoding='utf-8');
f.write("article_id,category\n")
for i in range(len(test_file)):
    f.write(str(i+1) + "," +  str(int(predicted[i])) + "\n");

f.close()