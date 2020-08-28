# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, os, functools, re

sys.path.append(os.getcwd() + '/..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from selenium import webdriver
import argparse

from random import random
from tqdm import tqdm
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

from langdetect import detect_langs
import nltk

from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

import seaborn as sns
import matplotlib.pyplot as plt

from source.utils import utils

CHECKPOINT_PATH = './baseline_best_weights/'
CHROME_PATH = '/home/vahidsanei_google_com/chromedriver/chromedriver'

class Data:
	def __init__(self, _url, _text, _label):
		self.url = _url
		self.text = _text
		self.label = _label
			
class BaseLineModel:
	def __init__(self, train, test, tokenizer: FullTokenizer, text_colname=None, 
		label_colname=None, max_seq_len=256, bert_config_file=None, bert_ckpt_file=None, is_multicase=True, classes=None, validation_split=0.2):
		""" 
			Prepare the date for training if the train set is given
			Args:
				max_seq_len: Shows the number of first words that are taken from the corpus. 
							 If the corpus lenght is smaller than max_seq_len, it is padded with zero.
				is_multicase: If True, the task is classification otherwise it is regression
		"""			
		self.tokenizer = tokenizer
		self.is_multicase = is_multicase

		try:
			nltk.data.find('tokenizers/punkt')
		except LookupError:
			nltk.download('punkt')
			os.system('python3 -m nltk.downloader stopwords')
				
		if train is not None:
			
			self.text_colname = 'webpage_corpus' if text_colname == None else text_colname
			if not self.text_colname in train.columns or not self.text_colname in test.columns:
				print('Error: Please specify a proper column name in the input dataframe as the corpus.')
				return

			self.label_colname = 'categories' if label_colname == None else label_colname
			if not self.label_colname in train.columns or not self.label_colname in test.columns:
				print('Error: Please specify a proper column name in the input dataframe as the labels.')
				return
				
			self.max_seq_len = 0
			
			self.classes = list(set(train[self.label_colname].unique().tolist() + test[self.label_colname].unique().tolist()))
			self.classes.sort()
			train = train.dropna(subset=[self.text_colname])
			test = test.dropna(subset=[self.text_colname])
			
			(self.train_x, self.train_y), (self.test_x, self.test_y) = map(functools.partial(self._tokanize), [train, test])
			self.max_seq_len = min(self.max_seq_len, max_seq_len)
			self.train_x, self.test_x = map(functools.partial(self._padding, max_seq_len=self.max_seq_len, with_cls_sep=True), [self.train_x, self.test_x])
			
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
			self.balanced_train_x, self.balanced_train_y = self._balance_classes(self.train_x, self.train_y)
		else:
			self.classes = classes
			if self.classes is not None: self.classes.sort()
			self.is_multicase = is_multicase
			self.max_seq_len = max_seq_len
			self.bert_model = self._load_bert(bert_config_file, bert_ckpt_file)

	def build_model(self, dropout=0.2, n_dense_layer=2, dense_size=768, is_multicase=True):
		"""
			Creates model that includes dense layers with dropouts. The input of this model
			is the first BERT embdedding of the input corpus.
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
		# take the first embedding of BERT as the output embedding
		output_ = keras.layers.Lambda(lambda seq: seq[:,0,:])(x)
		model = keras.Model(inputs=input_, outputs=output_)
		model.build(input_shape=(None, self.max_seq_len))
		load_stock_weights(bert, bert_ckpt_file)
		return model

	def _tokanize(self, df, max_sentence_len=None):
		X, y = [], []
		for _, entry in tqdm(df.iterrows()):
			corpus, label = entry[self.text_colname], entry[self.label_colname]
			corpus = utils.clean_text(corpus)
			tokens = self.tokenizer.tokenize(corpus)
			tokens = self._clean_tokens(tokens)

			if len(tokens) < 50:
				continue
				
			tokens = ['[CLS]'] + tokens + ['[SEP]']
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
		X = []
		for token_id in ids:
			id_ = self._cut_with_padding(token_id, self.max_seq_len)
			X.append(np.asarray(id_))
		return np.asarray(X)
	
	def _balance_classes(self, in_X, in_y):
		'''
			Down sample the data so that every class has the same number 
			of entries.
		'''
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
		
def predict_url(url_link, model, bm, max_len=256, chrome_path=CHROME_PATH):
	browser = webdriver.Chrome(executable_path=chrome_path)
	browser.get(url_link)
	text = browser.page_source
	text = utils.clean_text(text)
	tokens = bm.tokenizer.tokenize(text)
	ids = bm.tokenizer.convert_tokens_to_ids(tokens)
	ids = ids + [0 for _ in range(max_len  - len(ids))]
	ids = ids[:max_len]
	ids = np.squeeze(bm.bert_model.predict(np.asarray([ids])))
	ids = np.expand_dims(ids.reshape(-1), axis=0)
	probs = np.squeeze(model.predict(ids))
	return probs
    
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
	
	parser = argparse.ArgumentParser(
		description='Baseline -- Identifying Advertiser Quality from Their Websites'
	)
	parser.add_argument('--tasktype', help='(C) Classification or (R)Regression', choices=['C', 'R'])
	parser.add_argument('--input_directory', help='Directory for train and test data', default=None)
	parser.add_argument('--adam_lr', help='Adam learning rate', default=1e-5, type=float)
	parser.add_argument('--n_epochs', help='Numbrt of epochs', default=20, type=int)
	parser.add_argument('--val_split_ratio', help='Validation size ration', default=0.2, type=float)
	parser.add_argument('--bert_folder_path', help='Folder path of model BERT', default=os.path.join('/home/vahidsanei_google_com/', 'data','uncased_L-12_H-768_A-12'))
	parser.add_argument('--bert_embedding_size', help='BERT output embedding size', default=768, type=int)
	parser.add_argument('--keep_prob', help='Kept rate of dropout layers', default=0.6, type=float)
	parser.add_argument('--max_content_length', help='Maximum content length of from each leaf of DOM', default=256, type=int)
	parser.add_argument('--n_hidden_layers', help='Number of hidden layers', default=2, type=int)
	parser.add_argument('--url', help='URL link of business website', default=None)
	parser.add_argument('--best_weight_path', help='URL link of business website', default=None)
	parser.add_argument('--chrome_path', help='The path to chrome engine for Python package selenium', default=CHROME_PATH)
	
	st_args = utils.style()
	print(f'{st_args.BOLD}{st_args.RED}',end='')
	args = parser.parse_args()
	print(st_args.RESET, end='')
	
	tokenizer = FullTokenizer(vocab_file=os.path.join(args.bert_folder_path, 'vocab.txt'))
	bert_ckpt_file = os.path.join(args.bert_folder_path, 'bert_model.ckpt')
	bert_config_file = os.path.join(args.bert_folder_path, 'bert_config.json')
	
	if args.tasktype == 'C':
		is_multicase = True
		label_colname = 'categories'
		class_names = ['housework', 'health', 'financial', 'fitness', 'entertainment', 'car', 'law', 'food', 'beauty', 'education']
		class_names.sort()
	else: 
		is_multicase = False
		label_colname = 'stars'
		class_names = ['stars']
			
	if args.url is not None:
		if args.best_weight_path is None:
			best_weight_path = os.path.join(CHECKPOINT_PATH, label_colname)
		else:
			best_weight_path = args.best_weight_path
			
		with utils.suppress_stdout(), utils.suppress_sterr():
				bm = BaseLineModel(None, None, tokenizer, max_seq_len=args.max_content_length, bert_ckpt_file=bert_ckpt_file, bert_config_file=bert_config_file, 
					label_colname=label_colname, classes=class_names, is_multicase=is_multicase)
				model = bm.build_model(dropout=1.0 - (1e-8), n_dense_layer=args.n_hidden_layers, is_multicase=is_multicase)
				model.load_weights(best_weight_path)
				
		st = utils.style()
		if args.tasktype == 'C':	
			probs = predict_url(args.url, model, bm, max_len=args.max_content_length, chrome_path=args.chrome_path)
			ypred = int(np.argmax(probs))
			print(f' Predicted Class: {st.BOLD}{st.PURPLE}{class_names[ypred]}{st.RESET}, with probability {st.BOLD}{st.GREEN}{round(probs[ypred] * 100, 3)}%{st.RESET}')
		else:
			rating = predict_url(args.url, model, bm, max_len=args.max_content_length, chrome_path=args.chrome_path)
			print(f' Predicted Rating: {st.BOLD}{st.YELLOW}{rating:.2f}{st.RESET} out of 5')
	else:
		if args.input_directory is None:
			depth=4
			label_colname = 'categories' if is_multicase == True else 'stars'
			input_directory = f'/home/vahidsanei_google_com/data/yelp_data/oversampled_depth={depth}/{label_colname}/shuffle0'
		else:
			input_directory = args.input_directory
			
		input_df_train = pd.read_csv(os.path.join(input_directory, 'train.csv'))
		input_df_test = pd.read_csv(os.path.join(input_directory, 'test.csv'))
	
		bm = BaseLineModel(input_df_train, input_df_test, tokenizer, max_seq_len=args.max_content_length, bert_ckpt_file=bert_ckpt_file, bert_config_file=bert_config_file, 
				label_colname=label_colname, is_multicase=is_multicase)

		model = bm.build_model(dropout=args.keep_prob, n_dense_layer=args.n_hidden_layers, is_multicase=is_multicase)
		bm.compile_model(model, bm.train_x, bm.train_y, bm.val_x, bm.val_y, n_epochs=args.n_epochs, n_layers=args.n_hidden_layers, dropout=args.keep_prob, is_multicase=is_multicase)
		
		model.load_weights(CHECKPOINT_PATH)
		print('Accuracy on test set:')
		model.evaluate(bm.test_x, bm.test_y, return_dict=True)
