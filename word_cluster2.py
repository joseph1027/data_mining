import warnings
from pprint import pprint
import sys, codecs, gensim, logging, os, csv
import pandas as pd
import numpy as np
import re
import time
import gensim.models.keyedvectors as word2vec
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from numbers import Number
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
import _pickle as cPickle

def initial_csv():
	if os.path.exists('tmp.csv'):
		os.remove('tmp.csv')
	with open("Combined_News_DJIA.csv", "r") as f:
		with open('tmp.csv', 'a+', newline='\n') as csvfile:
			csvreader = csv.reader(f, delimiter=",")
			csvwriter = csv.writer(csvfile, delimiter=',')
			for line, content in enumerate(csvreader):
				# ignore first line
				if line == 0:
					print(content)
					continue;
				write_content = ""
				for i in range(10):
					word_list = re.sub(r'[\.|\!|\"|:|;|\?|\,|\'|\n|\(|\)|\[|\]|@]*', '', content[i+2])
					word_list = re.sub(r' +', ' ', word_list)
					word_list = word_list.replace("&amp",'').replace("\\r\\n",'').split(" ")
					for word in word_list:
						if word!='':
							write_content = write_content + " " + word.lower()
				csvwriter.writerow([content[1],write_content])

def initial_kmeans_model(k_clusters, train_x):
	clusterer = KMeans(n_clusters=k_clusters)
	cluster_labels = clusterer.fit_predict(train_x)
	#score the k classes
	silhouette_avg = silhouette_score(train_x, cluster_labels)
	print("For n_clusters =", k_clusters,
		  "The average silhouette_score is :", silhouette_avg)
	if os.path.exists('./model/kmeans_classifier.pkl'):
		os.remove('./model/kmeans_classifier.pkl')
	with open('./model/kmeans_classifier.pkl', 'wb') as fid:
		cPickle.dump(clusterer, fid) 
				
def build_sliding_window_data(row_data, column, window_size):
	np_array = row_data.as_matrix(columns=[column])
	n = np_array.strides[0]
	strided = np.lib.stride_tricks.as_strided
	return strided(np_array, shape=(np_array.shape[0]-window_size+1,window_size), strides=(n,n))
	
def build_unique_word2vec(google_model, row_data, start, end):
	word_index = dict()
	for line, words in enumerate(row_data):
		if line < start or line > end:
			continue
		word_list = words[1:].split(' ')
		for word in word_list:
			if word!='' and word not in word_index:
				try:
					vector = google_model[word]
					word_index[word] = vector
				except Exception as ex:
					continue
	return word_index
	
def build_word2vec(google_model, row_data, start, end):
	word_index = list()
	for line, words in enumerate(row_data):
		if line < start or line > end:
			continue
		long_sentence = words[0] + words[1]
		word_list = long_sentence.split(' ')
		row_word = []
		for word in word_list:
			if word!='':
				try:
					vector = google_model[word]
					row_word.append(vector)
				except Exception as ex:
					continue
		word_index.append(row_word)
	return word_index

def cluster_news_word(google_model, clusterer, row_data, start, end):
	news_word_vector = build_word2vec(google_model, row_data, start, end)
	classification = np.zeros((end-start+1,500), dtype=np.int)
	data_count = 0
	for daily_word_vector in news_word_vector:
		for word_vector in daily_word_vector:
			class_id = clusterer.predict(np.reshape(word_vector, (1,-1)))
			classification[data_count,class_id] += 1
		data_count += 1
	return classification

def error_count(predict,truth):
    predict = predict.ravel()
    truth = truth.ravel()
    total = max(predict.shape)
    hit = np.sum(predict == truth)
    return hit/total

def random_forest(x, label_data):
	clf = RandomForestClassifier(n_estimators=5, max_depth=10, random_state=0)
	clf.fit(x[0:1590,:], label_data[1:1591])
	predict_y = clf.predict(x[1591:-1,:])
	
	AC = error_count(predict_y,label_data[1592:])
	print(label_data[1592:])
	print(predict_y)
	print("random_forest")
	print(AC)
	
def naive_bayes(x, label_data):
	clf = GaussianNB()
	clf.fit(x[0:1590,:], label_data[1:1591])
	predict_y = clf.predict(x[1591:-1,:])
	
	AC = error_count(predict_y,label_data[1592:])
	print(label_data[1592:])
	print(predict_y)
	print("naive_bayes")
	print(AC)
	
def KNN(x, label_data):
	clf = KNeighborsClassifier(15)
	clf.fit(x[0:1590,:], label_data[1:1591])
	predict_y = clf.predict(x[1591:-1,:])
	
	AC = error_count(predict_y,label_data[1592:])
	print(label_data[1592:])
	print(predict_y)
	print("KNN")
	print(AC)
	
def SVM(x, label_data):
	clf = svm.SVC(kernel="rbf")
	clf.fit(x[0:1590,:], label_data[1:1591])
	predict_y = clf.predict(x[1591:-1,:])
	
	AC = error_count(predict_y,label_data[1592:])
	print(label_data[1592:])
	print(predict_y)
	print("SVM")
	print(AC)

def AdaBoost(x, label_data):
	clf = AdaBoostClassifier(n_estimators=15)
	clf.fit(x[0:1590,:], label_data[1:1591])
	predict_y = clf.predict(x[1591:-1,:])
	
	AC = error_count(predict_y,label_data[1592:])
	print(label_data[1592:])
	print(predict_y)
	print("AdaBoost")
	print(AC)
	
def main_fn():
	#initial_csv()
	column_names = ['label','news']
	daily_news = pd.read_csv('tmp.csv', header=None, names=column_names)
	daily_news = daily_news.reindex(index=daily_news.index[::-1])
	daily_news = daily_news.reset_index(drop=True)
	
	#import google words vector model
	train_limit = int(max(daily_news.shape))
	model = word2vec.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)  
	google_model = model.wv
	del model
	
	#turn news to vector
	word_vector = build_unique_word2vec(google_model,daily_news['news'], 0, train_limit)
	print(len(word_vector))
	
	train_x = [features for features in word_vector.values()]
	#initial_kmeans_model(500, train_x)
	with open('./model/kmeans_classifier.pkl', 'rb') as fid:
		clusterer = cPickle.load(fid)
	
	#build sliding window data
	news_data = build_sliding_window_data(daily_news, 'news', 2)
	label_data = build_sliding_window_data(daily_news, 'label', 2)[:,1]
	print(train_limit) #1989
	print(news_data.shape) #1987 * 3
	print(label_data.shape) #1987 * 1
	
	x = cluster_news_word(google_model, clusterer, news_data, 0, 1987)
	
	#SVM(x, label_data)
	#KNN(x, label_data)
	random_forest(x, label_data)
	naive_bayes(x, label_data)
	AdaBoost(x, label_data)

if __name__ == "__main__":
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		main_fn()
