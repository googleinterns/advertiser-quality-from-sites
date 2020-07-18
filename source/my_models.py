import sys
import os
import math
from random import random
import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import functools

import tensorflow as tf
from tensorflow import keras

from bs4 import BeautifulSoup
from bs4.element import Comment
import re
from langdetect import detect_langs
import nltk
from nltk.corpus import stopwords

from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from scipy.spatial import distance

class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

FOLDER_PATH = '/home/vahidsanei_google_com/'


class categoryDetection:    
	def __init__(self, train, test, tokenizer: FullTokenizer, text_colname=None, 
		label_colname=None, max_seq_len=128, glove_embeddings_address=None, word_similarity=False, glove_sentence_similarity=False,
		bert_sentence_similarity=False, bert_config_file=None, bert_ckpt_file=None, bert_pretrained_max_sent_len=None):
		"""  
		"""
		self.text_colname = 'webpage_corpus' if text_colname == None else text_colname
		if not self.text_colname in train.columns or not self.text_colname in test.columns:
			print('Error: Please specify a proper column name in the input dataframe as the corpus.')
			return

		self.label_colname = 'categories' if label_colname == None else label_colname
		if not self.label_colname in train.columns or not self.label_colname in test.columns:
			print('Error: Please specify a proper column name in the input dataframe as the labels.')
			return
		
		if word_similarity == True and glove_embeddings_address is None:
			print('Error: word_similarity is True while glove_embeddings_address is not provided.')
			return	

		if glove_sentence_similarity == True and glove_embeddings_address is None:
			print('Error: glove_sentence_similarity is True while glove_embeddings_address is not provided.')
			return
		
		if bert_sentence_similarity == True and (bert_config_file is None or bert_ckpt_file is None):
			print('Error: bert_sentence_similarity while either bert_config_file or bert_ckpt_file is not provided')
			return
			
		self.word_similarity = word_similarity
		self.glove_sentence_similarity = glove_sentence_similarity
		self.bert_sentence_similarity = bert_sentence_similarity
		self.max_seq_len = 0
		self.tokenizer = tokenizer
		self.classes = train[self.label_colname].unique().tolist()
		self.classes.sort()
		train = train.dropna(subset=[self.text_colname])
		test = test.dropna(subset=[self.text_colname])
				
		try:
			nltk.data.find('tokenizers/punkt')
		except LookupError:
			nltk.download('punkt')
			if sys.version_info > (3.0):
				os.system('python3 -m nltk.downloader stopwords')
			else:
				os.system('pyhton -m nltk.downloader.stopwords')
	
		self.glove_embeddings_address = glove_embeddings_address
		if self.glove_embeddings_address is not None:
			self.word2vec = self._load_glove(glove_embeddings_address)
			for cls in self.classes:
				if cls not in self.word2vec:
					print('Error: the embedding of {} is not provided'.format(cls))
					return
		if bert_sentence_similarity == True:
			(pretraintrain_x, pretraintrain_y) = self._prepare_for_pretraining(train, max_sentence_len=bert_pretrained_max_sent_len)
			'''
				A common plain English guideline says an average of 15–20 words 
				(Cutts, 2009; Plain English Campaign, 2015; Plain Language Association InterNational, 2015).
			'''
			self.pretrain_model = self.build_model(bert_config_file, bert_ckpt_file, max_seq_len=bert_pretrained_max_sent_len, n_dense_layer=1, adapter_size=None)
			print('Pretraining model ...')
			sys.stdout.flush()
			random_perm = np.random.permutation(len(pretraintrain_x))
			balanced_pretrain_x, balanced_pretrain_y = self._balance_classes(pretraintrain_x[random_perm], pretraintrain_y[random_perm])
			self.compile_model(self.pretrain_model, balanced_pretrain_x, balanced_pretrain_y, validation_split=0.0, batch_size=32, n_epochs=10)

		self.print_bert_sim_cnt = 0
		(self.train_x, self.train_y), (self.test_x, self.test_y) = map(functools.partial(self._tokanize, max_sentence_len=bert_pretrained_max_sent_len), [train, test])
		self.max_seq_len = min(self.max_seq_len, max_seq_len)
		self.train_x, self.test_x = map(functools.partial(self._padding, max_seq_len=self.max_seq_len, with_cls_sep=True), [self.train_x, self.test_x])
		self.balanced_train_x, self.balanced_train_y = self._balance_classes(self.train_x, self.train_y)

	def build_model(self, bert_config_file, bert_ckpt_file, max_seq_len, dropout=0.4, n_dense_layer=2, dense_size=800, adapter_size=64):
		"""
		"""
		bert = self._load_bert(bert_config_file, bert_ckpt_file, adapter_size)
		input_ = keras.layers.Input(shape=(max_seq_len, ), dtype='int64', name="input_ids")
		x = bert(input_)
		#get the first embedding from thde output of BERT
		x = keras.layers.Lambda(lambda seq: seq[:,0,:])(x)
		x = keras.layers.Dropout(dropout)(x)
		for _ in range(n_dense_layer):
			x = keras.layers.Dense(dense_size, activation='relu')(x)
			x = keras.layers.Dropout(dropout)(x)	
		output_ = keras.layers.Dense(units=len(self.classes), activation='softmax')(x)

		model = keras.Model(inputs=input_, outputs=output_)
		model.build(input_shape=(None, max_seq_len))

		load_stock_weights(bert, bert_ckpt_file)

		if adapter_size is not None:
			self._freeze_bert_layers(bert)

		return model

	def _load_bert(self, bert_config_file, bert_ckpt_file, adapter_size):
		try:
			with tf.io.gfile.GFile(bert_config_file, 'r') as gf:
				bert_config = StockBertConfig.from_json_string(gf.read())
				bert_params = map_stock_config_to_params(bert_config)
				bert_params.adapter_size = adapter_size
				bert = BertModelLayer.from_params(bert_params, name='bert')
				return bert
		except Exception as e:
			print(e)
			raise e

	def _flatten_layers(self, root_layer):
		if isinstance(root_layer, keras.layers.Layer):
			yield root_layer
		for layer in root_layer._layers:
			for sub_layer in self._flatten_layers(layer):
				yield sub_layer

	def _freeze_bert_layers(self, l_bert):
		"""
		Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
		"""
		for layer in self._flatten_layers(l_bert):
			if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
				layer.trainable = True
			elif len(layer._layers) == 0:
				layer.trainable = False
			l_bert.embeddings_layer.trainable = False

	def _load_glove(self, glove_address):
		try:
			file_path = os.path.join(glove_address)
		except FileNotFoundError:
			print('Glove embedding file is not found.')
			raise FileNotFoundError

		res = {}
		with open(file_path) as file:
			for s in file:
				arr = s.split()
				word = arr[0]
				embedding = np.asarray(arr[1:], dtype='float32')
				res[word] = embedding
		return res

	def _compute_glove_sentence_similarity(self, sentence) -> 'float32':
		words = sentence.split()
		sent_embedding = 0
		n_words = 0
		for word in words:
			if word in self.word2vec:
				sent_embedding += self.word2vec[word]
				n_words += 1
				
		if n_words == 0: 
			return -1.0
			
		sent_embedding = sent_embedding / n_words
		mx = 0
		for cls in self.classes:
			mx = max(mx, 1.0 - distance.cosine(sent_embedding, self.word2vec[cls]))
		return mx
		
	def _sort_sentences_with_glove_similarity(self, corpus):
		#print(corpus[:500])
		#print('*' * 100)
		sentences = corpus.split('#')
		sentence_with_similarity = []
		for sentence in sentences:
			sentence_with_similarity.append([sentence, self._compute_glove_sentence_similarity(sentence)])
		sentence_with_similarity = sorted(sentence_with_similarity, key=lambda x: x[1], reverse=True)
		
		sorted_sents = []
		for entry in sentence_with_similarity:
			tokens = entry[0].split()
			#if len(sorted_sents) == 0: tokens = ['CLS'] + tokens
			#tokens = tokens + ['[SEP]']
			sorted_sents.append(' '.join(tokens))
					
		sorted_corpus = ' '.join(sorted_sents)
		#print(sorted_corpus[:1000])
		#print('#' * 100)
		return sorted_corpus
		
	'''
		BERT SENTENCE SIMILARITY
	'''	
	def _compute_bert_sentence_similarity(self, sentence, max_sentence_len=50) -> 'float32':
		tokens = self.tokenizer.tokenize(sentence)
		id_ = self.tokenizer.convert_tokens_to_ids(tokens)
		id_ = self._cut_with_padding(id_, max_sentence_len, with_cls_sep=False)
		y_pred = self.pretrain_model.predict([id_])
		return np.max(y_pred)
	
	def _sort_sentences_with_bert_similarity(self, corpus, max_sentence_len=50):
		ran = random() if self.print_bert_sim_cnt < 5 else 1.0
		if self.print_bert_sim_cnt <= 5 and ran < 0.05:
			print(corpus[:1000])
			print('*' * 100)
		sentences = corpus.split('#')
		sentence_with_similarity = []
		for sentence in sentences:
			sentence_with_similarity.append([sentence, self._compute_bert_sentence_similarity(sentence, max_sentence_len=max_sentence_len)])
		sentence_with_similarity = sorted(sentence_with_similarity, key=lambda x: x[1], reverse=True)
		
		sorted_sents = []
		for entry in sentence_with_similarity:
			tokens = entry[0].split()
			#if len(sorted_sents) == 0: tokens = ['CLS'] + tokens
			#tokens = tokens + ['[SEP]']
			sorted_sents.append(' '.join(tokens))
					
		sorted_corpus = ' '.join(sorted_sents)
		
		if self.print_bert_sim_cnt <= 5 and ran < 0.05:
			print(sorted_corpus[:1000])
			print('#' * 100)
			print('#' * 100)
			self.print_bert_sim_cnt += 1
		return sorted_corpus

	def _prepare_for_pretraining(self, df, max_sentence_len=50):
		print('Preparing data for the pretraining step ...')
		sys.stdout.flush()
		X, y = [], []
		for _, entry in tqdm(df.iterrows()):
			corpus, label = entry[self.text_colname], entry[self.label_colname]
			sentences = corpus.split('#')			
			for sentence in sentences:
				tokens = self.tokenizer.tokenize(sentence)
				id_ = self.tokenizer.convert_tokens_to_ids(tokens)
				id_ = self._cut_with_padding(id_, max_sentence_len, with_cls_sep=False)
				X.append(np.array(id_))
				y.append(self.classes.index(label))		
		return np.asarray(X), np.asarray(y)
		
	def _word_similarity(self, word):
		if word not in self.word2vec: return 0
		word_embedding = self.word2vec[word]
		mx = 0
		for cls in self.classes:
			mx = max(mx, 1.0 - distance.cosine(word_embedding, self.word2vec[cls]))
		return mx

	def _tokanize(self, df, max_sentence_len=None):
		"""
		"""
		X, y = [], []
		for _, entry in tqdm(df.iterrows()):
			corpus, label = entry[self.text_colname], entry[self.label_colname]

			if self.glove_sentence_similarity == True:
				corpus = self._sort_sentences_with_glove_similarity(corpus)
				
			if self.bert_sentence_similarity == True:
				corpus = self._sort_sentences_with_bert_similarity(corpus, max_sentence_len=max_sentence_len)
				
			tokens = self.tokenizer.tokenize(corpus)
			tokens = self._clean_tokens(tokens)

			if len(tokens) <= 50:
				continue
				
			if self.word_similarity == True:
				tokens = self._sort_words_with_similarity(tokens)
			else:
				tokens = ['[CLS]'] + tokens + ['[SEP]']
				
			#all_tokens.append(tokens)
			ids = self.tokenizer.convert_tokens_to_ids(tokens)
			self.max_seq_len = max(self.max_seq_len, len(ids))
			X.append(ids)
			y.append(self.classes.index(label))

		print('Removed {}% of entries, due to being short corpus length.'.format((1.0 - len(X) / len(df)) * 100.0))

		return np.asarray(X), np.asarray(y)

	def _sort_words_with_similarity(self, tokens):
		words_with_similarity = []
		for token in tokens:
			sim = self._word_similarity(token)
			words_with_similarity.append([token, sim])
			
		words_with_similarity = sorted(words_with_similarity, key=lambda x: x[1], reverse=True)
		tokens = [entry[0] for entry in words_with_similarity]
		return tokens

	def _clean_tokens(self, tokens):
		# STOPS = set(stopwords.words('english'))
		clean_tokens = []
		for token in tokens:
			if any(map(str.isdigit, token)): 
				continue
			clean_tokens.append(token)
		return clean_tokens
		
	def _cut_with_padding(self, token_id, max_seq_len, with_cls_sep=True):
		if with_cls_sep == True:
			CLS_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
			SEP_id = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
		if with_cls_sep == True:
			arr = token_id[1:-1]
			sz = min(len(arr), max_seq_len - 2)
			arr = CLS_id + arr[:sz] + SEP_id
		else:
			arr = token_id
			sz = min(len(arr), max_seq_len)
			arr = arr[:sz]
		# pad the remaining cells with zero
		arr = arr + [0] * (max_seq_len - len(arr))
		return arr
				
	def _padding(self, ids, max_seq_len, with_cls_sep=True):
		"""
		"""
		X = []
		for token_id in ids:
			id_ = self._cut_with_padding(token_id, self.max_seq_len)
			X.append(np.asarray(id_))
		return np.asarray(X)
	
	def _balance_classes(self, in_X, in_y):
		count = [0 for _ in range(len(self.classes))]
		for label in in_y:
			count[label] += 1
		mn = len(in_X)
		for i in range(len(count)):
			mn = min(mn, count[i])
		print('size of each balanced class = ', mn)
		count = [0 for _ in range(len(self.classes))]
		X, y = [], []
		for tokens, label in zip(in_X, in_y):
			assert count[label] <= mn, 'count is greater than mn!'
			if count[label] == mn: continue
			count[label] += 1
			X.append(tokens)
			y.append(label)
		return np.asarray(X), np.asarray(y)


	def compile_model(self, model, train_x, train_y, validation_split=0.05, batch_size=16, n_epochs=30, shuffle=True):
		#log_dir = "/home/wliang_google_com/Documents/workspace/notebook/.log/website_rating/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
		#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
		model.compile(optimizer=keras.optimizers.Adam(1e-5),
		  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		  metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')])
		print(model.summary())
		history = model.fit(
			x=train_x,
			y=train_y,
			validation_split=validation_split,
			batch_size=batch_size,
			shuffle=shuffle,
			verbose=1,
			epochs=n_epochs,
		)
	
def unify_yelp_data_classes(df, map_classes = {
                'Restaurants': 'Food', 'Food': 'Food', 'Frozen Yogurt': 'Food', 'Pizza': 'Food', 'Bars': 'Food', 'Coffee': 'Food',
                'Cafes': 'Food', 'Fast Food': 'Food', 'Bakeries': 'Food', 'Tea' : 'Food', 'Breakfast': 'Food',
                'Wine': 'Food', 'Sandwiches': 'Food', 'Burgers': 'Food', 'Brunch': 'Food', 'Breakfast': 'Food', 'Desserts': 'Food',
                'Vegetarian': 'Food', 'Vegan': 'Food', 

                'Health': 'Health', 'Dentists': 'Health', 'Doctors': 'Health', 'Medical Centers': 'Health', 'Drugstores': 'Health', 

                'Car Dealers': 'Car', 'Automotive': 'Car', 'Auto Repair': 'Car',
                 
                'Home Services': 'Housework','Garden': 'Housework',
                'Pet Services': 'Housework', 'Home Cleaning': 'Housework', 'Laundry': 'Housework', 'Laundry Services': 'Housework',
                'Home Decor': 'Housework', 'Pets': 'Housework', 'Carpet Cleaning': 'Housework',
    
                'Hair Salons': 'Beauty', 'Nail Salons': 'Beauty', 'Beauty': 'Beauty', 'Hair Salons': 'Beauty', 'Makeup Artists': 'Beauty',
                'Hair Removal': 'Beauty', 'Massage': 'Beauty', 'Barbers': 'Beauty', 'Beauty Supply': 'Beauty',
                
                'Entertainment': 'Entertainment', 
                'Active Life': 'Entertainment', 'Nightlife': 'Entertainment',
                #'Hotels': 'Entertainment', 'Travel': 'Entertainment', 
                'Hobby Shops': 'Entertainment', 

                'Fitness': 'Fitness', 'Sporting Goods': 'Fitness', 'Gyms': 'Fitness', 'Sports Bars': 'Fitness', 'Golf': 'Fitness',
                
                'Education': 'Education',
                
                'DUI Law': 'Law', 'Lawyers': 'Law', 'Real Estate': 'Law', 'Real Estate Law': 'Law', 'Divorce': 'Law',
            
                'Banks': 'Financial', 'Financial Services': 'Financial',
    
                #'Mass Media': 'Entertainment',
                #'Churches': 'Religious', 'Religious': 'Religious', 'Religious Organizations': 'Religious'
        }
    , show_skipped=False):
		
	df = df[df['categories'].notnull()]
	df['categories'] = df['categories'].apply(lambda x: re.split('[,;&]', x))
	
	if show_skipped == True:
		show_skipped_classes(df, map_classes)
		return df
		
	cat = []
	for arr in df['categories']:
		cat.append(None)
		majority_vote = {}
		for x in arr:
			cls = x.strip()
			if not cls in map_classes:
				continue
			y_str = map_classes[cls].lower()
			if y_str not in majority_vote: majority_vote[y_str] = 0
			else: majority_vote[y_str] += 1
		if len(majority_vote) != 0:
			cat[-1] = max(majority_vote, key=lambda k: majority_vote.get(k))
	df['categories'] = cat
	df = df[df['categories'].notnull()]
	if show_skipped == True:
		show_skipped_classes(df, map_classes)
	return df

def show_skipped_classes(df, map_classes):
	#This function is to check which entires are removed due to not belonging to any defined classes
	cat = {}
	bad = []
	for x in df.categories:
		flg = False
		for cls in x:
			cls = cls.strip()
			if not cls in map_classes: continue
			flg = True
			mapped_cls = map_classes[cls]
			if mapped_cls not in cat: cat[mapped_cls]=1
			else: cat[mapped_cls]+=1
		if flg is False:
			bad.append(x)
			
	print(bad[:50], '\n', len(bad))
	
def get_classes_distribution(train_y, classes):
	count = {}
	for y in train_y:
		if y not in count: count[y] = 0
		count[y] += 1
	for y in np.unique(train_y):
		print(f'class {classes[y]} = {round(count[y] / len(train_y) * 100.0, 2)}%')
		
def testing(text, trained_model, cat:categoryDetection, LEN=300):
    tokens = cat.tokenizer.tokenize(text)
    ids = cat.tokenizer.convert_tokens_to_ids(tokens)
    ids = ids + [0 for _ in range(LEN  - len(ids))]
    ids = ids[:LEN]
    y_pred = trained_model.predict([ids])
    print(cat.classes[np.argmax(y_pred)])
    prediction = [f'{round(y_ * 100, 2)}%' for y_ in y_pred[0]]
    print(list(zip(prediction, cat.classes)))
    
def find_wrongs(cat, model, count=20):
    cnt = 0
    for ids, label in zip(cat.test_x, cat.test_y):
        if cnt == count:
            break
        y_pred = np.argmax(model.predict(np.asarray([ids])))
        if y_pred != label:
            corpus = cat.tokenizer.convert_ids_to_tokens(ids)
            print(' '.join(corpus), '\n', f'{style.GREEN}label = {cat.classes[label]} {style.RED}predicted = {cat.classes[y_pred]}{style.RESET}')
            print('*' * 100)
            cnt += 1    
def plot_classes_distribution(df, class_name='categories'):
	chart = sns.countplot(df[class_name], palette='hls')
	plt.title('Number of business per Category')
	chart.set_xticklabels(chart.get_xticklabels(), horizontalalignment='center', rotation=90)
	plt.show();

if __name__ == '__main__':
	# for testing
	df_file = os.path.join(FOLDER_PATH, 'data', 'yelp_data', 'updated_with_sent', 'business_with_corpus.csv')
	glove_file = os.path.join(FOLDER_PATH, 'data', 'glove_data', 'glove.6B.300d.txt')

	df = pd.read_csv(df_file)
	df = df[:10]
	
	folder_path = os.path.join(FOLDER_PATH, 'data','uncased_L-12_H-768_A-12')
	tokenizer = FullTokenizer(vocab_file=os.path.join(folder_path, 'vocab.txt'))

	bert_ckpt_file = os.path.join(folder_path, 'bert_model.ckpt')
	bert_config_file = os.path.join(folder_path, 'bert_config.json')

	print(df.info())
	
	df = unify_yelp_data_classes(df)
	
	sz = int(len(df) * 0.8)
	train = df[:sz]
	test = df[sz:]
	
	cat = categoryDetection(train, test, tokenizer, max_seq_len=300, glove_embeddings_address=glove_file, bert_sentence_similarity=True, bert_ckpt_file=bert_ckpt_file, bert_config_file=bert_config_file)

	
