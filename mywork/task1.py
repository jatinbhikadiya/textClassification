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
'''
import os
import string
import time
from collections import Counter
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir,"data")
#from nltk.stem import PorterStemmer
def read_data(deals_file = os.path.join(data_dir,"deals.txt")):
    
    ''' This function reads the data from deals.txt and performs all 
    pre-prosessing. It removes punctuations, stop words and lematizes
    the words. Also null lines in the file are removed'''
    
      
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
            deals.append(deal)
        
    print len(deals)
    f.close()
    print time.time()-start_time
    f_write = open(os.path.join(data_dir,"deals_new"),'w')
    for deal in deals:
        f_write.write("%s\n"%deal)
    f_write.close()
    return deals


def word_counts(deals):
    '''Given list of deals, this function finds the word frequency and 
    answers first three question'''
    cnt = Counter()
    for deal in deals:
        for word in deal.split():
            cnt[word] +=1
    f_write = open(os.path.join(data_dir,"word_counts.txt"),'w')
    n=len(cnt)
    for word,frequency in cnt.most_common(n):
        f_write.write("%s %d\n"%(word,frequency))
    f_write.close()
    print "Most Popular term : ",cnt.most_common(n)[0]
    print "Least Popular term :",cnt.most_common(n)[-1]
    
def types_of_guitar(deals):
    guitar_deals = []
    for deal in deals:
        for word in deal.split():
            if word=='guitar':
                guitar_deals.append(deal)
                break
    pos_tags = []
    for deal in guitar_deals:
        tag = pos_tag(word_tokenize(deal))
        pos_tags.append(tag)
    guitar_types = []
    for tag,deal in zip(pos_tags,guitar_deals):
        indices = [i for i,x in enumerate(deal.split()) if x=='guitar']
        for i in indices:
            if i>0 and tag[i-1][1]=='JJ' and tag[i-1][0] not in guitar_types:
                guitar_types.append(tag[i-1][0])
    print "types of guitar :"
    for type in guitar_types:
        print type
    guitar_file = os.path.join(data_dir,"guitar_types.txt")
    
 
if __name__ == '__main__':
    deals = read_data()
    word_counts(deals)
    types_of_guitar(deals)
