""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""
'''This program uses gensim package for the latent sematic analysis'''

import os
import string
import time
import numpy as np
import sys
from collections import Counter
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
import gensim
from gensim import corpora, models, similarities

from nltk.stem import WordNetLemmatizer


import scipy.cluster.hierarchy as hcluster


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir,"data")


def read_data():
    
    ''' This function reads the data from deals.txt and performs all 
    pre-prosessing. It removes punctuations, stop words and lematizes
    the words. Also null lines in the file are removed'''
    print "Preprocessing the given corpus...\n"
    deals_file = os.path.join(data_dir,"deals.txt")   
    stop_words_file = os.path.join(data_dir,"stop_words.txt")
    
    start_time = time.time() 
    f = open(deals_file,'r')
    f_stop_words = open(stop_words_file,'r')
    stop_words = [word.strip() for word in f_stop_words]
    stop_words = set(stop_words)
    
    char_map = string.maketrans("","")
    wnl = WordNetLemmatizer()
    #p = PorterStemmer()

    deals = []
    for line in f:
        '''remove punctuations'''
        deal = line.strip()
        if ".com" not in deal:
            deal= deal.translate(char_map,string.punctuation).lower()
        else:
            deal= deal.translate(char_map,
                                 string.punctuation.replace(".", "")).lower()
        ''' remove emply lines and stop words'''
        if deal != '':
            '''removes stop words'''
            deal = " ".join(word for word in 
                            deal.split() if word not in stop_words)
            deal =  " ".join((wnl.lemmatize(word)) for word in deal.split())
            deal = [word for word in deal.split()]
            deals.append(deal)
    f.close()
    f_write = open(os.path.join(data_dir,"task_2_0_deals_new.txt"),'w')
    for deal in deals:
        f_write.write("%s\n"%deal)
    f_write.close()
    print "Preprocessing done. Preprocessed deals are writen to ../data/task_2_0_deals_new.txt\n"

    return deals


def create_dictionary(deals):
    '''Given list of deals, this function finds the word frequency and 
    amd usomf corpora class from gensim, it creates a dictionary of tokens.
    Dictionary is saves in the 'data' folder'''
    tokens = sum(deals,[])
    tokens_less_frequent = set(word for word in set(tokens) if tokens.count(word)<4)
    deals = [[word for word in deal if word not in tokens_less_frequent] for deal in deals]
    print "Creating dictionary\n"
    dictionary = corpora.Dictionary(deals)
    dictionary.save(os.path.join(data_dir,"task_2_0_dictionary.dict"))
    print "Dictionary generation done\n"

def extract_topics(deals):
    '''This function takes deals, dictionary name(assuming its already inside
    the data folder. It generates the bag of words version of all the tokenized
    deals. ALso it runs the LSI model on the corpus to find the 200 topics.
    Topics are listed in the file "task_2_0_topic_list_lsi.txt" insise the 
    data directory '''
    dictionary = corpora.Dictionary.load(os.path.join(data_dir,"task_2_0_dictionary.dict"))
    #print dictionary.token2id
    corpus = [dictionary.doc2bow(deal) for deal in  deals]
    corpora.MmCorpus.serialize(os.path.join(data_dir,"task_2_0_lsi_corpus.mm"),corpus)
    '''if corpus is already on the disk'''
    #corpus = gensim.corpora.MmCorpus(os.path.join(data_dir,"task_2_0_lsi_corpus.mm"))
    print "Extracting topics"
    lsi = gensim.models.lsimodel.LsiModel(corpus=corpus,id2word=dictionary,num_topics=200)
    lsi.save(os.path.join(data_dir,"lsi_model.model"))
    print "Topics extraction done"
    lsi.print_topics(10)
    topics = lsi.show_topics(num_topics=-1, num_words=10, log=False, formatted=True)
    
    f_write = open(os.path.join(data_dir,"task_2_0_topic_list_lsi.txt"),'w')
    for topic in topics:
        f_write.write("%s\n"%topic)
    f_write.close()
    print "Topics are written to 'data/task_2_0_topic_list_lsi.txt'"

def main():
	deals = read_data()
	create_dictionary(deals)
	extract_topics(deals)
	

if __name__ == '__main__':
    main()
