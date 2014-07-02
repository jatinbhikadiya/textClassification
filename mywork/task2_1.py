""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""
import os
import string
import time
from collections import Counter
import numpy as np
from nltk.stem import WordNetLemmatizer
parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir,"data")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

#from nltk.stem import PorterStemmer
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
            deals.append(deal)
        
    f.close()
    print time.time()-start_time
    f_write = open(os.path.join(data_dir,"task_2_1_deals_new.txt"),'w')
    for deal in deals:
        f_write.write("%s\n"%deal)
    f_write.close()
    print "Preprocessing done. Preprocessed deals are writen to ../data/task_2_1_deals_new.txt\n"
    return deals
    

def get_word_counts(deals,write=0):
    '''Given list of deals, this function finds the word frequency '''
    cnt = Counter()
    for deal in deals:
        for word in deal.split():
            cnt[word] +=1
    if write:
        f_write = open(os.path.join(data_dir,"task_2_1_counts.txt"),'w')
        n=len(cnt)
        for word,frequency in cnt.most_common(n):
            f_write.write("%s %d\n"%(word,frequency))
        f_write.close()
        
    return cnt
    
            
def groups_with_given_dict(deals,word_counts,sparse='True'):
    word_with_low_freq = [word for word in word_counts.elements() if word_counts[word]<15]
    for word in word_with_low_freq:
        del word_counts[word]
    
    tfidf_vectorizer = TfidfVectorizer(vocabulary=word_counts)
    tfidf = tfidf_vectorizer.fit_transform(deals)
    if not sparse:
		tfidf = tfidf.todense()
        
    t0 = time.time()

    '''Kmeans with sklearn. I ran it on for n_jobs=-2, which will break
     down matrix into 14 slices and compute them in parallel'''
    km = KMeans(n_clusters=40 , init='k-means++', max_iter=2, tol=0.0001,
                 precompute_distances=True, verbose=0, random_state=None,
                 copy_x=True, n_jobs=1)
    print "Have cluster input (#no_of_deals, #dimensionality):", tfidf.shape
    print "Starting k-means with %d iterations to find %d clusters" % (2, 40)
    km.fit(tfidf)
    clusters = km.cluster_centers_
    labels = km.labels_
    print("done in %0.3fs" % (time.time() - t0))
    np.save(os.path.join(data_dir,"clusters_2_1_sparse"),clusters)
    np.save(os.path.join(data_dir,"labels_2_1_sparse"),labels)
    deals_groups = get_groups(deals,labels,40)
    save_groups_to_file(deals_groups)


def get_groups(deals,labels,number_of_clusters):
    '''This function will extract 40 group of deals based on the
    clustering labels. Given deals, labels and number of cluster,
    it will return the list of deals for each cluster
    If we have deals file(tokenized) and labels stored as np array
    on disk than it can be directly used.'''
    
    #deals_file = os.path.join(data_dir,"task_2_1_deals_new.txt")   
    #labels = np.load(os.path.join(data_dir,"labels_2_1_sparse.npy"))
    #deals = []
    #f = open(deals_file,'r')
    #for line in f:
    #    deals.append(line.strip())
    
    indices = np.argsort(labels)
    counts = []
    for i in range(40):
        counts.append(labels.tolist().count(i))
    counts.reverse()
    counts.append(0)
    counts.reverse()
    grouplist= [[]]
    for i in range(1,len(counts)):
        start_index = sum(counts[0:i])
        end_index= sum(counts[0:i+1])
        deal_indices = indices[start_index:end_index]
        group = []
        for index in deal_indices:
            group.append(deals[index])
        grouplist.append(group)
    grouplist.pop(0)
    return grouplist

def save_groups_to_file(groups_list):
    '''In this function, I am extracting import keywords in each groups.
    Also, each group is saved with the keywords at top in the 'data/groups'
    folder.'''
    
    group_dir = os.path.join(data_dir,'groups')
    if not os.path.exists(group_dir):
        os.mkdir(group_dir)
    for i in range(len(groups_list)):
        deals_within_group = groups_list[i]
        dictionary = get_word_counts(deals_within_group)
        popular_words = sorted(dictionary, key = dictionary.get, reverse = True)
        n = min(10,len(dictionary))
        top_n = popular_words[:n]
        f_write = open(os.path.join(group_dir,'group_'+str(i)+'sparse'+'.txt'),'w')
        f_write.write(" Popular words in group : ")
        for word in top_n:
            f_write.write("%s\t"%word)
        for deal in deals_within_group:
            f_write.write("%s\n"%deal)
        f_write.close()

def main():
	deals = read_data()
	keyword_count = get_word_counts(deals,1)
	groups_with_given_dict(deals,keyword_count)

	   
    
if __name__ == '__main__':
	main()

    
