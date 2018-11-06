import pandas as pd
from bs4 import BeautifulSoup
import re, unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import numpy as np
import nltk
from sklearn.pipeline import Pipeline
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.moses import MosesDetokenizer
from itertools import product

## read file
train_file = pd.read_csv("train_v2.csv");
start=0
end=len(train_file)#int(len(train_file)/10*7)
print (end,len(train_file))
train_category = train_file.loc[start:end,'category'];

##################
## pre-processing - https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html
## Functions below are all pre processing steps
# remove html text
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

# normalization of text
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    return words

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

#######
## implementation of the pre-processing functions.
## remove html for title
title_col = train_file.loc[start:end,'title'];
publisher_col = train_file.loc[start:end,'publisher'];
combined_str = ["" for i in range(len(title_col))];
for i in range(len(title_col)):
    # combine the title and publisher string
    combined_str[i] = str(title_col[i]) + " " + str(publisher_col[i]);
    combined_str[i] = denoise_text(combined_str[i]);

for i in range(len(combined_str)):
    # combine the title and publisher string
#     print ("[Before Processing] ",combined_str[i])
    combined_str[i] = nltk.word_tokenize(combined_str[i]);
    combined_str[i] = normalize(combined_str[i]);
    combined_str[i] = lemmatize_verbs(combined_str[i])
    for j in range(len(combined_str[i])):
        if len(combined_str[i][j])<=2:
            combined_str[i][j] = "";
    detokenizer = MosesDetokenizer()
    combined_str[i] = detokenizer.detokenize(combined_str[i], return_str=True)
#     print ("[After Processing] ",combined_str[i],"\n")


##################
## Count vectorize and transform string to values
tcount_vect = CountVectorizer(token_pattern=r'\b[^\d\W]+\b') # exclude numbers
combined_str_counts = tcount_vect.fit_transform(combined_str)
ttfidf_transformer = TfidfTransformer()
title_train_tfidf = ttfidf_transformer.fit_transform(combined_str_counts)
print (len(title_train_tfidf.toarray()[0]),len(train_category))

# part to extract adj and noun (not improving after using this.)
extract_adj_noun_index = []
for i in range(len(tcount_vect.get_feature_names())):
    found=False;
    for syn in wn.synsets(tcount_vect.get_feature_names()[i]): 
        if syn.pos() in ['a', 's', 'v']:
            found=True;
    if found:
        extract_adj_noun_index.append(i)

fnames = np.array(tcount_vect.get_feature_names());
print ("list of adj and noun", fnames[extract_adj_noun_index])

##########
# not using this part
# from sklearn.feature_selection import SelectPercentile
# from sklearn.feature_selection import chi2
# percentile_select = SelectPercentile(chi2, percentile=10);
# X_new = percentile_select.fit_transform(combined_str_counts, train_category)
# fnames = np.array(tcount_vect.get_feature_names())
# print (fnames[np.where(percentile_select.get_support())[0]]);

# for i in range(len(np.where(percentile_select.get_support())[0])):
#     print (fnames[np.where(percentile_select.get_support())[0]][i] , "\t" , np.sum(combined_str_counts[:,i]))

# f = open("test.csv","w");
# f.write("title\tcategory\t")
# for i in range(len(X_new[0])):
#     f.write(fnames[np.where(percentile_select.get_support())[0]][i] + "\t");
# f.write("\n");
# for i in range(len(title_train_tfidf.toarray())):
#     pos0 = np.where(percentile_select.get_support())[0][0]
#     f.write(combined_str[i] + "\t" + str(train_category[i]) + "\t" + str(title_train_tfidf.toarray()[i][pos0]));
#     for j in range(1,len(np.where(percentile_select.get_support())[0])):
#         pos = np.where(percentile_select.get_support())[0][j]
#         f.write("\t" + str(title_train_tfidf.toarray()[i][pos]))
#     f.write("\n");
# f.close();

######################
print ("[Number of features] ", len(tcount_vect.get_feature_names()))

# from sklearn.feature_selection import SelectPercentile
# from sklearn.feature_selection import chi2
# percentile_select = SelectPercentile(chi2, percentile=50);
# X_new = percentile_select.fit_transform(title_train_tfidf.toarray(), train_category)
# print (X_new.shape)

## grid search to get the best possible parameters
# parameters = {'n_neighbors':[5,10,15,20,25,30,35], 'n_estimators':[50,100]}
parameters = {'loss':["hinge","squared_hinge"],'C':[0.05,0.5,1.0]};
ens = LinearSVC(class_weight="balanced"); #GradientBoostingClassifier(n_estimators=10, max_depth=5)
clf = GridSearchCV(ens, parameters, cv=5)
print ("training....")
clf.fit(title_train_tfidf.toarray(), train_category) # [:,extract_adj_noun_index]
prd = clf.predict(title_train_tfidf.toarray() # [:,extract_adj_noun_index])
print("best estimator: " ,clf.best_estimator_, "best score: " ,clf.best_score_)

print (np.sum(prd==train_category)/float(len(train_category)))
print (np.column_stack((combined_str,prd)))

## test csv file (ignore)
# categories = {0:"Ratings downgrade", 1:"Sanctions", 2:"Growth into new markets", 3:"New product coverage", 4:"Others"};
# f = open("test2.csv",'w', encoding='utf-8');
# f.write("article_id,category\n")
# for i in range(len(train_category)):
#     f.write(str(combined_str[i]) + "," +  str(categories.get(int(prd[i]))) + "\n");
# f.close()


###############################
## test case for submission
test_file = pd.read_csv('test_v2.csv');
test_title_col = test_file.loc[start:end,'title'];
test_publisher_col = test_file.loc[start:end,'publisher'];
# print (test_title_col, test_publisher_col)
test_combined_str = ["" for i in range(len(test_title_col))];
for i in range(len(test_file)):
    # combine the title and publisher string
#     print (i,test_title_col[i], test_publisher_col[i])
    test_combined_str[i] = str(test_title_col[i]) + " " + str(test_publisher_col[i]);
    test_combined_str[i] = denoise_text(test_combined_str[i]);

for i in range(len(test_combined_str)):
    # combine the title and publisher string
#     print ("[Before Processing] ",combined_str[i])
    test_combined_str[i] = nltk.word_tokenize(test_combined_str[i]);
    test_combined_str[i] = normalize(test_combined_str[i]);
    test_combined_str[i] = lemmatize_verbs(test_combined_str[i])
    detokenizer = MosesDetokenizer()
    test_combined_str[i] = detokenizer.detokenize(test_combined_str[i], return_str=True)

test_combined_str_counts = tcount_vect.transform(test_combined_str)
test_title_train_tfidf = ttfidf_transformer.transform(test_combined_str_counts)

# X_new_test = percentile_select.transform(test_title_train_tfidf.toarray())
# print (X_new_test.shape)

predicted = clf.predict(test_title_train_tfidf.toarray());# [:,extract_adj_noun_index]

categories = {0:"Ratings downgrade", 1:"Sanctions", 2:"Growth into new markets", 3:"New product coverage", 4:"Others"};
for i in range(len(predicted)):
    print (test_title_col[i], "\t", categories.get(predicted[i]))
    
f = open("Submission_LinearSVC_squared_hinge_C0.05.csv",'w', encoding='utf-8');
f.write("article_id,category\n")
for i in range(len(test_file)):
    f.write(str(i+1) + "," +  str(int(predicted[i])) + "\n");

f.close()