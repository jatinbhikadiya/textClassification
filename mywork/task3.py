""" Classification

The objective of this task is to build a classifier that can tell us whether a new, unseen deal 
requires a coupon code or not. 

We would like to see a couple of steps:

1. You should use bad_deals.txt and good_deals.txt as training data
2. You should use test_deals.txt to test your code
3. Each time you tune your code, please commit that change so we can see your tuning over time

Also, provide comments on:
    - How general is your classifier?
    - How did you test your classifier?

"""
'''Classification problem can be solved two ways:
1) Sentiment Analysis -  There are not really good words or bad words in 
the deals which will help in the sentiment analysis
2) TFIDF vector.-  I have performed classification using TFIDF vector.
And from my previous experiments, I have concluded that SVM performs better 
with the TFIDF vector(Bag of words representation)

To generalize the classifier, model estimator from the scikit is used. 
Grid of gamma values, C values and Kernel is givne and with the grid search 
the best parameters are selected.

Also the number of features for the tfidf representaion is important. 
Best score achieved is 71.66% classification rate.

The test_deals are tested on the trained classifier and results are saved to
"data/predicted_test_deals.txt" with predicted labels.
'''
import os
import re, math, collections, itertools
import time
import numpy as np
from scikits.learn import svm, grid_search
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir,"data")

def read_data():
    
    ''' This function reads the data from deals.txt and performs all 
    pre-prosessing. It removes punctuations, stop words and lematizes
    the words. Also null lines in the file are removed'''
    
    good_deals_file = os.path.join(data_dir,"good_deals.txt")
    bad_deals_file = os.path.join(data_dir,"bad_deals.txt")   
    stop_words_file = os.path.join(data_dir,"stop_words.txt")
    test_deals_file = os.path.join(data_dir,"test_deals.txt")
    f_stop_words = open(stop_words_file,'r')
    stop_words = [word.strip() for word in f_stop_words]
    stop_words.append("com")
    stop_words = set(stop_words)
    
    
    wnl = WordNetLemmatizer()
    start_time = time.time() 
    f = open(good_deals_file,'r')
    good_deals = []     
    for line in f:
        deal = re.findall(r"[\w']+|[!?;]%", line)
        '''removes stop words'''
        deal = " ".join(word for word in 
                        deal if word not in stop_words)
        deal =  " ".join((wnl.lemmatize(word)) for word in deal.split())
        good_deals.append(deal)
    f.close()
    
    f = open(bad_deals_file,'r')
    bad_deals = []     
    for line in f:
        deal = re.findall(r"[\w']+|[%!?;]", line)
        '''removes stop words'''
        deal = " ".join(word for word in 
                        deal if word not in stop_words)
        deal =  " ".join((wnl.lemmatize(word)) for word in deal.split())
        bad_deals.append(deal)
        
    f.close()
    
    f = open(test_deals_file,'r')
    test_deals = []     
    for line in f:
        deal = re.findall(r"[\w']+|[!?;]%", line)
        '''removes stop words'''
        deal = " ".join(word for word in 
                        deal if word not in stop_words)
        deal =  " ".join((wnl.lemmatize(word)) for word in deal.split())
        test_deals.append(deal)
    f.close()
    
    
    return [good_deals,bad_deals,test_deals]

def word_counts(deals):
    '''Given list of deals, this function finds the word frequency and 
    answers first three question'''
    cnt = Counter()
    for deal in deals:
        for word in deal.split():
            cnt[word] +=1
    return cnt

def classify(good_deals,bad_deals,dictionary):
    word_with_low_freq = [word for word in dictionary.elements() if dictionary[word]<1]
    for word in word_with_low_freq:
        del dictionary[word]
    
    tfidf_vectorizer = TfidfVectorizer(vocabulary=dictionary)
    good_tfidf = tfidf_vectorizer.fit_transform(good_deals)
    bad_tfidf = tfidf_vectorizer.fit_transform(bad_deals)
    good_tfidf = good_tfidf.todense()
    bad_tfidf = bad_tfidf.todense()
    svm_data = []
    svm_data.append(good_tfidf)
    svm_data.append(bad_tfidf)
    svm_data = np.concatenate(svm_data)
    svm_pos_lables = np.ones(len(good_tfidf))
    svm_neg_lables = np.zeros(len(bad_tfidf))
    labels= []
    labels.append(svm_pos_lables)
    labels.append(svm_neg_lables)
    svm_labels  = np.concatenate(labels)
    
    param_grid = [
                  {'C': [1, 10, 100, 1000], 'gamma': [1,0.1,0.001, 0.0001],'kernel': ['linear']},
                  {'C': [1, 10, 100, 1000], 'gamma': [1,0.1,0.001, 0.0001], 'kernel': ['rbf']},
                  ]
    svc = svm.SVC()
    clf = grid_search.GridSearchCV(estimator=svc, param_grid=param_grid,n_jobs=1)
    print "Training SVM classifier for grid of C and gamma values to select best parameter\n"
    clf.fit(svm_data,svm_labels)
    print "svm score",clf.best_score
    print "svm gamma value",clf.best_estimator.gamma
    print "svm C value",clf.best_estimator.C
    print "svm kernel",clf.best_estimator.kernel
    return clf

def test_data(test_deals,classifier,dictionary):
    word_with_low_freq = [word for word in dictionary.elements() if dictionary[word]<1]
    for word in word_with_low_freq:
        del dictionary[word]
    
    tfidf_vectorizer = TfidfVectorizer(vocabulary=dictionary)
    test_tfidf = tfidf_vectorizer.fit_transform(test_deals)
    test_tfidf = test_tfidf.todense()
    predictions  = classifier.predict(test_tfidf)
    test_deals_file = os.path.join(data_dir,"test_deals.txt")
    predicted_test_deals_file = os.path.join(data_dir,"predicted_test_deals.txt")
    
    f=open(test_deals_file,'r')
    f_write = open(predicted_test_deals_file,'w')

    for deal,predicted_label in zip(f,predictions):
        if predicted_label==0:
            f_write.write("Bad deal -- %s"%deal)
        else :
            f_write.write("Good deal -- %s"%deal) 
    f.close()
    f_write.close()
    print "Predicted labels with test deals are written to 'data/predicted_test_deals.txt'"

if __name__ == '__main__':
    [good_deals,bad_deals,test_deals] = read_data()
    dictionary = word_counts(bad_deals+good_deals)
    classifier = classify(good_deals,bad_deals,dictionary)
    test_data(test_deals,classifier,dictionary)
