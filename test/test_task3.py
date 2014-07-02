""" Features

The objective of this task is to explore the corpus, deals.txt. 

The deals.txt file is a collection of deal descriptions, separated by a new line, from which 
we want to glean the following insights:

1. What is the most popular term across all the deals?
2. What is the least popular term across all the deals?
3. How many types of guitars are mentioned across all the deals?

"""
'''
Created on Apr 4, 2014

@author: bhikadiy

Task 3 takes good and bad deals. It trains SVM classifier using the 
tfidf vectors. The labels for the test deals are predicted and 
stored in the file "data/predicted_test_deals.txt"
'''
import os
parent_dir = os.path.dirname(os.getcwd())
import sys
sys.path.append(parent_dir)
from mywork import task3
print "Reading good,bad and test deals"
[good_deals,bad_deals,test_deals] = task3.read_data()
dictionary = task3.word_counts(bad_deals+good_deals)
print "Training classifier"
classifier = task3.classify(good_deals,bad_deals,dictionary)
print "Predicting labels for test deals"
task3.test_data(test_deals,classifier,dictionary)
