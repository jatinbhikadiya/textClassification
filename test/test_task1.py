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
3) Types of guitar
'''
import os
parent_dir = os.path.dirname(os.getcwd())
import sys
sys.path.append(parent_dir)
from mywork import task1
'''Read data returns the deals after preprocessing. It takes the  filename
as an argument. If no argument is given then it reads deals from the ../data/deals.txt'''
deals = task1.read_data()

'''word_counts takes deals as an argument and returns the most populat
and least popular word. The word_counts are stored in the file data/word_counts.txt  '''
task1.word_counts(deals)

'''types_of_guitar takes deals and prints the type of guitar extracted using pos tagging. 
Also, it saves guitar types in the file data/guitar_types.txt'''
task1.types_of_guitar(deals)
