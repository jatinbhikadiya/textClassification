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

Task 1 contains three functions. 1) To read data, 2) get frequency of the words 
3) Generates groups.
The groups are saved to the direcotry data/group/<group_no>.txt

There are two version based on the matrix used for the clustering. Dense or sparse matrix can be
used for clustering. If sparse matrix will be used then the clustering will be done in nearly 
10 seconds using 16 core machine.

Task2_0 extracts the topics . Topics are saved to data/task_2_0_topic_list_lsi.txt
'''
import os
parent_dir = os.path.dirname(os.getcwd())
import sys
sys.path.append(parent_dir)
from mywork import task2_1,task2_0
print "Generating groups"

deals = task2_1.read_data()
keyword_count = task2_1.get_word_counts(deals,1)
task2_1.groups_with_given_dict(deals,keyword_count)

print "Extracting topics"
deals = task2_0.read_data()
'''Creating dictionary takes around 50-60 minutes. It is already stored
inside the data directory. So no need to run it every time.
'''
#task2_0.create_dictionary(deals) 
task2_0.extract_topics(deals)
