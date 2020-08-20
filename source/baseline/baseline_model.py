import sys, os, math, datetime, functools, re
sys.path.append(os.getcwd() + '/..')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from random import random
from tqdm import tqdm
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

from bs4 import BeautifulSoup
from bs4.element import Comment
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

from utils import utils
from utils import process_websites

FOLDER_PATH = '/home/vahidsanei_google_com/'
CHECKPOINT_PATH = './baseline_best_weights/'

class Data:
	def __init__(self, _url, _text, _label):
		self.url = _url
		self.text = _text
		self.label = _label
			
class BaseLineModel:
		
	def __init__(self, train, test, tokenizer: FullTokenizer, text_colname=None, 
		label_colname=None, max_seq_len=256, bert_config_file=None, bert_ckpt_file=None, is_multicase=True, validation_split=0.2):
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
			
		self.max_seq_len = 0
		self.tokenizer = tokenizer
		self.classes = list(set(train[self.label_colname].unique().tolist() + test[self.label_colname].unique().tolist()))
		self.classes.sort()
		self.is_multicase = is_multicase
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

		(self.train_x, self.train_y), (self.test_x, self.test_y) = map(functools.partial(self._tokanize), [train, test])
		self.max_seq_len = min(self.max_seq_len, max_seq_len)
		self.train_x, self.test_x = map(functools.partial(self._padding, max_seq_len=self.max_seq_len, with_cls_sep=True), [self.train_x, self.test_x])
		
		##
		self.bert_model = self._load_bert(bert_config_file, bert_ckpt_file)
		self.train_x, self.test_x = map(self._get_bert_embedding, [self.train_x, self.test_x])
		
		self.best_val_loss=float('inf')
		self.best_dropout=-1
		self.best_n_layers=-1
		
		# take validation from test as we have already oversampled training set.												
		split_size = int(len(self.test_x) * (1.0 - validation_split))
		self.val_x = self.test_x[split_size:]
		self.val_y = self.test_y[split_size:]
		self.test_x = self.test_x[:split_size]
		self.test_y = self.test_y[:split_size]
		
		# this is for down sampling
		#self.balanced_train_x, self.balanced_train_y = self._balance_classes(self.train_x, self.train_y)

	def build_model(self, dropout=0.2, n_dense_layer=2, dense_size=768, is_multicase=True):
		"""
		"""
		if is_multicase == True:
			output_classes = len(self.classes)
			output_activation = 'softmax'
		else:
			output_classes = 1
			output_activation = 'linear'
			
		input_ = keras.layers.Input(shape=(768, ), dtype='float32', name="input_ids")
		x = input_
		for _ in range(n_dense_layer):
			x = keras.layers.Dense(dense_size, activation='relu')(x)
			x = keras.layers.Dropout(dropout)(x)	
		
		output_ = keras.layers.Dense(units=output_classes, activation=output_activation)(x)
		model = keras.Model(inputs=input_, outputs=output_)
		model.build(input_shape=(None, 768))		
		return model
		
	def _get_bert_embedding(self, X):
		res = []
		sys.stdout.write('Get embeddings:\n')
		for x in tqdm(X):
			res.append(np.squeeze(self.bert_model.predict(np.asarray([x]))))
		return np.asarray(res)
		
	def _load_bert(self, bert_config_file, bert_ckpt_file):
		try:
			with tf.io.gfile.GFile(bert_config_file, 'r') as gf:
				bert_config = StockBertConfig.from_json_string(gf.read())
				bert_params = map_stock_config_to_params(bert_config)
				bert_params.adapter_size = None
				bert = BertModelLayer.from_params(bert_params, name='bert')
		except Exception as e:
			print(e)
			raise e
			
		input_ = keras.layers.Input(shape=(self.max_seq_len, ), dtype='int64', name="input_ids")
		x = bert(input_)
		output_ = keras.layers.Lambda(lambda seq: seq[:,0,:])(x)
		model = keras.Model(inputs=input_, outputs=output_)
		model.build(input_shape=(None, self.max_seq_len))
		load_stock_weights(bert, bert_ckpt_file)
		return model

	def _tokanize(self, df, max_sentence_len=None):
		"""
		"""
		X, y = [], []
		for _, entry in tqdm(df.iterrows()):
			corpus, label = entry[self.text_colname], entry[self.label_colname]
			corpus = utils.clean_text(corpus)
			tokens = self.tokenizer.tokenize(corpus)
			tokens = self._clean_tokens(tokens)

			if len(tokens) < 50:
				continue
				
			tokens = ['[CLS]'] + tokens + ['[SEP]']
				
			#all_tokens.append(tokens)
			ids = self.tokenizer.convert_tokens_to_ids(tokens)
			self.max_seq_len = max(self.max_seq_len, len(ids))
			X.append(ids)
			if self.is_multicase == True:
				y.append(self.classes.index(label))
			else:
				y.append(label)

		print('Removed {}% of entries, due to being short corpus length.'.format((1.0 - len(X) / len(df)) * 100.0))

		return np.asarray(X), np.asarray(y)

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

	def compile_model(self, model, train_x, train_y, val_x, val_y, batch_size=16, n_epochs=30, shuffle=True, dropout=None, n_layers=None, is_multicase=True):
		#log_dir = "/home/wliang_google_com/Documents/workspace/notebook/.log/website_rating/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
		#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

		model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, 
																	monitor='val_loss', mode='min', save_best_only=True)																	

		if is_multicase == True:
			model.compile(optimizer=keras.optimizers.Adam(1e-5),
			  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			  metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')])
		else:
			model.compile(optimizer=keras.optimizers.Adam(1e-5),
				loss=tf.keras.losses.MeanSquaredError(),
				metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse'), tf.keras.metrics.MeanAbsoluteError(name='mae')])
		  
		print(model.summary())
		
		history = model.fit(
			x=train_x,
			y=train_y,
			validation_data=(val_x, val_y),
			batch_size=batch_size,
			shuffle=shuffle,
			verbose=1,
			epochs=n_epochs,
			callbacks=[model_checkpoint_callback]
		)
		
		res = model.evaluate(val_x, val_y, return_dict=True)
		if res['loss'] < self.best_val_loss:
			self.best_dropout = dropout
			self.best_n_layers = n_layers
			self.best_val_loss = res['loss']
		
def testing(text, trained_model, cat, LEN=300):
    tokens = cat.tokenizer.tokenize(text)
    ids = cat.tokenizer.convert_tokens_to_ids(tokens)
    ids = ids + [0 for _ in range(LEN  - len(ids))]
    ids = ids[:LEN]
    y_pred = trained_model.predict([ids])
    sys.stdout.write(f'Predicted class: {cat.classes[np.argmax(y_pred)]}\n')
    prediction = [f'{round(y_ * 100, 3)}%' for y_ in y_pred[0]]
    sys.stdout.write(list(zip(prediction, cat.classes)) + '\n')
    sys.stdout.flush()
    
def find_wrongs(cat, model, count=20):
	cnt = 0
	test_X, test_y = cat._tokanize(cat)
	for X, y, entry in zip(test_x, test_y, cat):
		if cnt == count: break
		y_pred = np.argmax(model.predict(X))
		if y_pred != label:
			sys.stdout.write(entry.text, '\n', f'{style.GREEN}label = {cat.classes[label]} {style.RED}predicted = {cat.classes[y_pred]}{style.RESET}\n')
			sys.stdout.write(('*' * 100) + '\n')
			cnt += 1

if __name__ == '__main__':
	# for testing
	label_colname='categories'
	is_multicase = True if label_colname == 'categories' else False
	
	train = pd.read_csv(os.path.join(FOLDER_PATH, 'data', 'yelp_data', 'oversampled_depth=6', label_colname, 'shuffle0', 'train.csv'))
	test = pd.read_csv(os.path.join(FOLDER_PATH, 'data', 'yelp_data', 'oversampled_depth=6', label_colname, 'shuffle0', 'test.csv'))
	
	#~ train = train[:200]
	#~ test = test[:200]
	
	#~ glove_file = os.path.join(FOLDER_PATH, 'data', 'glove_data', 'glove.6B.300d.txt')
	folder_path = os.path.join(FOLDER_PATH, 'data','uncased_L-12_H-768_A-12')
	tokenizer = FullTokenizer(vocab_file=os.path.join(folder_path, 'vocab.txt'))
	bert_ckpt_file = os.path.join(folder_path, 'bert_model.ckpt')
	bert_config_file = os.path.join(folder_path, 'bert_config.json')
	#
	
	cat = BaseLineModel(train, test, tokenizer, max_seq_len=256, bert_ckpt_file=bert_ckpt_file, bert_config_file=bert_config_file, label_colname=label_colname, is_multicase=is_multicase)
	
	## search for the best dropout and #hidden layers based on the result of validation
	#~ for n_l in [2, 3, 4, 5]:
		#~ for dp_out in [0.1, 0.2, 0.3, 0.4, 0.5]:
			#~ model = cat.build_model(dropout=dp_out, n_dense_layer=n_l, is_multicase=is_multicase)
			#~ cat.compile_model(model, cat.train_x, cat.train_y, n_epochs=30, n_layers=n_l, dropout=dp_out, is_multicase=is_multicase)
			#~ break
		#~ break
		
		
	#~ print(f'The best droupout={cat.best_dropout}, best hidden layer={cat.best_n_layers}')
	#~ model = cat.build_model(dropout=cat.best_dropout, n_dense_layer=cat.best_n_layers, is_multicase=is_multicase)
	model = cat.build_model(dropout=0.6, n_dense_layer=5, is_multicase=is_multicase)
	cat.compile_model(model, cat.train_x, cat.train_y, cat.val_x, cat.val_y, n_epochs=100, n_layers=cat.best_n_layers, dropout=cat.best_dropout, is_multicase=is_multicase)
	model.load_weights(CHECKPOINT_PATH)
	
	print('Accuracy on test set:')
	res = model.evaluate(cat.test_x, cat.test_y, return_dict=True)
	y_pred = model.predict(cat.test_x)
	#~ print(y_pred[:100])
