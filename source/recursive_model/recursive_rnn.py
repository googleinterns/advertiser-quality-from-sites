import sys, os, math, time, itertools, shutil, random
sys.path.append(os.getcwd() + '/..')

import numpy as np
import pandas as pd
from tqdm import tqdm
import socket

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf
import tensorflow as tftwo
from tensorflow import keras
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

import tree_lib
from utils import utils


def overrides(interface_class):
    def overrider(func):
        assert (func.__name__ in dir(interface_class)) is True
        return func
    return overrider

class Configuration():
	def __init__(self, input_df_train=None, input_df_test=None, lr=1e-2, n_epochs=40, l2=0.001, val_split_ratio=None, anneal_factor=1.5, 
			anneal_tolerance=0.99, patience_early_stopping=5, max_depth=None, n_classes=None, bert_folder_path=None, 
			bert_embedding_size=768, embedding_size=768, keep_prob=0.65, max_content_length=256, verbose=1):
		assert input_df_train is not None, 'input_df_train cannot be None'
		assert input_df_test is not None, 'input_df_train cannot be None'
		assert n_classes is not None, '# of classes must be specified'
		assert bert_folder_path is not None, 'input files for BERT must be specified'

		self.input_df_train = input_df_train
		self.input_df_test = input_df_test
		self.lr = lr
		self.n_epochs = n_epochs
		self.l2 = l2
		self.val_split_ratio = val_split_ratio
		self.anneal_factor = anneal_factor
		self.anneal_tolerance = anneal_tolerance
		self.patience_early_stopping = patience_early_stopping
		self.max_depth = max_depth
		self.n_classes = n_classes
		
		self.bert_folder_path = bert_folder_path
		self.bert_ckpt_file = os.path.join(self.bert_folder_path, 'bert_model.ckpt')
		self.bert_config_file = os.path.join(self.bert_folder_path, 'bert_config.json')
		self.tokenizer = FullTokenizer(vocab_file=os.path.join(self.bert_folder_path, 'vocab.txt'))
		# the output size of uncased_L-12_H-768_A-12 is 768
		self.bert_embedding_size = bert_embedding_size
		self.embedding_size = embedding_size
		self.keep_prob = keep_prob
		self.max_content_length = max_content_length
		self.verbose = verbose

		# save weights after set value steps, to remove the constructed trees from RAM (to avoid OOM)
		self.AVOID_OOM_AT = 100
		self.model_name = None
	
class RecursiveModel():
	def __init__(self, configuration):		
		
		self.configuration = configuration
		self.__set_model_name__()				
		training_trees = tree_lib.read_trees(self.configuration.input_df_train, verbose=self.configuration.verbose)
		self.test_trees = tree_lib.read_trees(self.configuration.input_df_test, verbose=self.configuration.verbose)
			
		self.bert_model = self.load_bert()
		if self.configuration.verbose > 0:
			sys.stdout.write(' [Preprocessing step] Get embeddings of leaves contents:\n')
			sys.stdout.flush()
			
		for tree in (tqdm(training_trees) if self.configuration.verbose > 0 else training_trees):
			self.__content_to_embedding__(tree.root)
		for tree in (tqdm(self.test_trees) if self.configuration.verbose > 0 else self.test_trees):
			self.__content_to_embedding__(tree.root)
					
		tf.disable_v2_behavior()
		#tf.disable_eager_execution()
		
		self.val_trees = None
		if self.configuration.val_split_ratio is not None:
			split_size = int((1.0 - self.configuration.val_split_ratio) * len(training_trees))
			self.val_trees = training_trees[split_size:]
			self.train_trees = training_trees[:split_size]
		else:
			self.train_trees = training_trees[:]
			
		if self.configuration.verbose > 0:
			sys.stdout.write(f' Training data distribution with {len(self.train_trees)} entries\n')
			sys.stdout.flush()
			self.__cls_distributions__(self.train_trees)
			
			if self.val_trees is not None:
				sys.stdout.write(f' Validation data distribution with {len(self.val_trees)} entries\n')
				sys.stdout.flush()
				self.__cls_distributions__(self.val_trees)
			
			sys.stdout.write(f' Test data distribution with {len(self.test_trees)} entries\n')
			sys.stdout.flush()
			self.__cls_distributions__(self.test_trees)
			sys.stdout.write('*' * 100 + '\n')
			sys.stdout.flush()			
			
		self.weights_path = os.path.join('.', 'weights', f'{self.configuration.model_name}.ckpt')
		self.best_weights_path = os.path.join('.', 'weights', f'best.{self.configuration.model_name}.ckpt')
		
	def __set_model_name__(self):
		self.configuration.model_name = f'{socket.gethostname()}_{self.__class__.__name__}_cls={self.configuration.n_classes}'
		
	def __cls_distributions__(self, trees):
		cls_count = {}
		for tree in trees:
			if tree.label not in cls_count: cls_count[tree.label] = 1
			else: cls_count[tree.label] += 1
		
		total = sum(cls_count.values())
		cls_count = dict(sorted(cls_count.items()))
		for key, value in cls_count.items():
			sys.stdout.write(f' Class {key} has {round(value / total * 100, 2)}% of entries.\n')
		sys.stdout.flush()
		
	def __content_to_embedding__(self, node):
		if node.is_leaf == True:
			node.bert_embedding = np.asarray(self.bert_model.predict([self.__cut_with_padding__(node.content)]))
		else:
			for child in node.children:
				self.__content_to_embedding__(child)
	
	def __cut_with_padding__(self, content):
		tokens = self.configuration.tokenizer.tokenize(content)
		ids = self.configuration.tokenizer.convert_tokens_to_ids(tokens)
		# pad the remaining cells with zero
		ids = ids + [0 for _ in range(self.configuration.max_content_length - len(ids))]
		ids = ids[:self.configuration.max_content_length]
		return ids
	
	def load_bert(self):
		try:
			with tftwo.io.gfile.GFile(self.configuration.bert_config_file, 'r') as gf:
				bert_config = StockBertConfig.from_json_string(gf.read())
				bert_params = map_stock_config_to_params(bert_config)
				bert_params.adapter_size = None
				bert = BertModelLayer.from_params(bert_params, name='bert')
		except Exception as e:
			print(e)
			raise e
			
		input_ = keras.layers.Input(shape=(self.configuration.max_content_length, ), dtype='int64', name="input_ids")
		x = bert(input_)
		output_ = tf.squeeze(keras.layers.Lambda(lambda seq: seq[:,0,:])(x))
		model = keras.Model(inputs=input_, outputs=output_)
		model.build(input_shape=(None, self.configuration.max_content_length))
		load_stock_weights(bert, self.configuration.bert_ckpt_file)
		return model
		
	def get_tensor(self, node):			
		if node.is_leaf == True:
			with tf.variable_scope('bert_layer', reuse=True):
				W_bert = tf.get_variable('W_bert')
				b_bert = tf.get_variable('b_bert')
			node_tensor = tf.nn.relu(tf.add(tf.matmul(node.bert_embedding, W_bert), b_bert))
		else:	
			children_tensors_list = []
			for child in node.children:
				children_tensors_list.append(self.get_tensor(child))
			children_tensors = tf.concat(children_tensors_list, 1)
							
			with tf.variable_scope(f'layer{len(node.children)}', reuse=True):
				W = tf.get_variable(f'W{len(node.children)}')
				b = tf.get_variable(f'b{len(node.children)}')
			node_tensor = tf.nn.relu(tf.add(tf.matmul(children_tensors, W), b))
			
		node_tensor = tf.nn.dropout(node_tensor, rate=1.0 - self.configuration.keep_prob)									
		return node_tensor
	
	def _compute_logit(self, tree):
		root_tensor = self.get_tensor(tree.root)
		with tf.variable_scope('output_layer', reuse=True):
			W = tf.get_variable('W0')
			b = tf.get_variable('b0')
		logit = tf.add(tf.matmul(root_tensor, W), b)
		return logit
		
	def get_variables(self):
		with tf.variable_scope('output_layer'):
			tf.get_variable('W0', [self.configuration.embedding_size, self.configuration.n_classes])
			tf.get_variable('b0', [1, self.configuration.n_classes])
			
		for d in range(1, self.configuration.max_depth + 1):
			with tf.variable_scope(f'layer{d}'):
				tf.get_variable(f'W{d}', [self.configuration.embedding_size * d, self.configuration.embedding_size])
				tf.get_variable(f'b{d}', [1, self.configuration.embedding_size])

		with tf.variable_scope('bert_layer'):
			tf.get_variable('W_bert', [self.configuration.bert_embedding_size, self.configuration.embedding_size])
			tf.get_variable('b_bert', [1, self.configuration.embedding_size])
	
	def _compute_loss(self, label, logit):
		cross_entropy_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=tf.constant(label)))
				
		Ws = []
		with tf.variable_scope('output_layer', reuse=True):
			W = tf.get_variable('W0')
			Ws.append(W)
		for d in range(1, self.configuration.max_depth + 1):
			with tf.variable_scope(f'layer{d}', reuse=True):
				W = tf.get_variable(f'W{d}')
				Ws.append(W)
		with tf.variable_scope('bert_layer', reuse=True):
			W = tf.get_variable('W_bert')
			Ws.append(W)
			
		total_weight_loss = tf.reduce_sum([tf.nn.l2_loss(W) for W in Ws])
		total_loss = self.configuration.l2 * total_weight_loss + cross_entropy_loss
		return total_loss
	
	def tree_loss(self, tree, logit=None):
		if logit is None: logit = self._compute_logit(tree)
		return self._compute_loss([tree.label], logit)
		
	def predict(self, trees, weights_path):
		predictions, losses = [], []
		pos = 0
		if self.configuration.verbose > 0:
			sys.stdout.write(' Prediction:\n')
			sys.stdout.flush()
			rogbar = keras.utils.Progbar(len(trees))
		while pos < len(trees):
			with tf.Graph().as_default(), tf.Session() as sess:
				self.get_variables()
				saver = tf.train.Saver()
				saver.restore(sess, weights_path)
				for _ in range(self.configuration.AVOID_OOM_AT):
					if pos >= len(trees): break
					
					tree = trees[pos]
					logit = self._compute_logit(tree)
					tf_prediction = tf.argmax(logit, 1)
					y_pred = sess.run(tf_prediction)[0]
					predictions.append(y_pred)
					tree_loss = sess.run(self.tree_loss(tree, logit=logit))
					losses.append(tree_loss)
					
					pos += 1
					if self.configuration.verbose > 0: 
						rogbar.update(pos)
		return predictions, losses
		
	def run_epoch(self, epoch_number):
		random.shuffle(self.train_trees)
		losses = []
		pos = 0
		if not os.path.exists('./weights'):
			os.makedirs('./weights')
		while pos < len(self.train_trees):
			with tf.Graph().as_default(), tf.Session() as sess:
				self.get_variables()
				if epoch_number == 0 and pos == 0:
					sess.run(tf.global_variables_initializer())
				else:
					saver = tf.train.Saver()
					saver.restore(sess, self.weights_path)
				
				for i in range(self.configuration.AVOID_OOM_AT):
					if pos >= len(self.train_trees): break
		
					tree = self.train_trees[pos]
					loss = self.tree_loss(tree)
					optimizer = tf.train.GradientDescentOptimizer(self.configuration.lr).minimize(loss)
					l, _ = sess.run([loss, optimizer])
					losses.append(l)
					
					if self.configuration.verbose > 0:
						sys.stdout.write(f'\r Epoch:{epoch_number + 1}/{self.configuration.n_epochs}, {pos + 1}/{len(self.train_trees)}: loss={np.mean(losses)}')
						sys.stdout.flush()
					pos += 1
					
				saver = tf.train.Saver()
				saver.save(sess, self.weights_path)
		
		if self.configuration.verbose > 0:
			sys.stdout.write('\n')
			sys.stdout.flush()		
		
		train_predictions, train_losses = self.predict(self.train_trees, weights_path=self.weights_path)
		if self.val_trees is None:
			val_predictions, val_losses = None, None
		else:
			val_predictions, val_losses = self.predict(self.val_trees, weights_path=self.weights_path)
		
		train_labels = [t.label for t in self.train_trees]
		val_labels = None if self.val_trees is None else [t.label for t in self.val_trees]
		train_accuracies = np.equal(train_labels, train_predictions)
		val_accuracies = None if self.val_trees is None else np.equal(val_labels, val_predictions)
		
		return train_accuracies, val_accuracies, train_losses, val_losses
			
	def train_model(self):
		train_losses = []
		train_accuracies = []
		val_losses = []
		val_accuracies = []
		last_epoch_loss = float('inf')
		best_val_acc = -1
		best_val_epoch = -1
		
		for epoch in range(self.configuration.n_epochs):
			start_time = time.time()		
			epoch_train_accuracies, epoch_val_accuracies, epoch_train_losses, epoch_val_losses = self.run_epoch(epoch_number=epoch)
			epoch_time = time.time() - start_time
			train_acc, train_loss = np.mean(epoch_train_accuracies), np.mean(epoch_train_losses)
			train_accuracies.append(train_acc)	
			train_losses.append(train_loss)
			
			if epoch_val_accuracies is not None:
				val_acc, val_loss =  np.mean(epoch_val_accuracies), np.mean(epoch_val_losses)
				val_losses.append(val_loss)
				val_accuracies.append(val_acc)	
				sys.stdout.write(f' Train acc = {round(train_acc * 100, 3)}%, Val acc = {round(val_acc * 100, 3)}%, Train loss = {train_loss}, Val loss = {val_loss}, lr={self.configuration.lr}, epoch time={round(epoch_time, 2)}s\n')
			else:
				sys.stdout.write(f' Train acc = {round(train_acc * 100, 3)}%, Train loss = {train_loss}, lr={self.configuration.lr}, epoch time={round(epoch_time, 2)}s\n')
			sys.stdout.write(('*' * 150) + '\n')
			sys.stdout.flush()		

			if train_losses[-1] > last_epoch_loss * self.configuration.anneal_tolerance:
				self.configuration.lr /= self.configuration.anneal_factor
			
			# HERE my random/stupid ideas arise
			#~ if epoch_val_accuracies is not None:
				#~ if val_loss * 0.98 > train_loss:
					#~ self.configuration.l2 = min(0.03, self.configuration.l2 * 1.15)
				#~ elif train_losses[-1] > last_epoch_loss * self.configuration.anneal_tolerance:
					#~ self.configuration.l2 = max(0.001, self.configuration.l2 / 1.15)
					
			last_epoch_loss = train_losses[-1]
			
			
			if epoch_val_accuracies is not None and val_acc > best_val_acc:
					best_val_acc = val_acc
					best_val_epoch = epoch
					shutil.copyfile(f'{self.weights_path}.data-00000-of-00001', f'{self.best_weights_path}.data-00000-of-00001')
					shutil.copyfile(f'{self.weights_path}.index', f'{self.best_weights_path}.index')
					shutil.copyfile(f'{self.weights_path}.meta', f'{self.best_weights_path}.meta')
			
			#if epoch - best_val_epoch > self.configuration.patience_early_stopping:
			#	TODO: stop training.
			
	def evaluate_testset(self):
		test_predictions, test_losses = self.predict(self.test_trees, self.best_weights_path)
		test_labels = [t.label for t in self.test_trees]
		test_acc = np.mean(np.equal(test_labels, test_predictions))
		sys.stdout.write(f' Test acc = {round(test_acc * 100, 3)}%\n')

class RecursiveModelWithLessParams(RecursiveModel):
	def __init__(self, configuration):
		super().__init__(configuration)
	
	@overrides(RecursiveModel)
	def get_tensor(self, node):			
		if node.is_leaf == True:
			with tf.variable_scope('bert_layer', reuse=True):
				W_bert = tf.get_variable('W_bert')
				b_bert = tf.get_variable('b_bert')
			node_tensor = tf.nn.relu(tf.matmul(node.bert_embedding, W_bert) + b_bert)
		else:	
			with tf.variable_scope(f'internal_layer', reuse=True):
				W = tf.get_variable(f'W_internal')
				b = tf.get_variable(f'b_internal')

			for idx, child in enumerate(node.children):
				if idx == 0: node_tensor = self.get_tensor(child)
				else:
					children_tensors = tf.concat([node_tensor, self.get_tensor(child)], 1)
					node_tensor = tf.nn.relu(tf.matmul(children_tensors, W) + b)
					
		node_tensor = tf.nn.dropout(node_tensor, rate=1.0 - self.configuration.keep_prob)									
		return node_tensor
	
	@overrides(RecursiveModel)
	def _compute_loss(self, label, logit):
		cross_entropy_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=tf.constant(label)))
		
		Ws = []
		with tf.variable_scope('output_layer', reuse=True):
			W = tf.get_variable('W0')
			Ws.append(W)

		with tf.variable_scope(f'internal_layer', reuse=True):
			W = tf.get_variable(f'W_internal')
			Ws.append(W)
			
		with tf.variable_scope('bert_layer', reuse=True):
			W = tf.get_variable('W_bert')
			Ws.append(W)

		total_weight_loss = tf.reduce_sum([tf.nn.l2_loss(W) for W in Ws])
		total_loss = self.configuration.l2 * total_weight_loss + cross_entropy_loss
		return total_loss
	
	@overrides(RecursiveModel)
	def get_variables(self):
		with tf.variable_scope('output_layer'):
			tf.get_variable('W0', [self.configuration.embedding_size, self.configuration.n_classes])
			tf.get_variable('b0', [1, self.configuration.n_classes])
			
		with tf.variable_scope(f'internal_layer'):
			tf.get_variable(f'W_internal', [self.configuration.embedding_size * 2, self.configuration.embedding_size])
			tf.get_variable(f'b_internal', [1, self.configuration.embedding_size])

		with tf.variable_scope('bert_layer'):
			tf.get_variable('W_bert', [self.configuration.bert_embedding_size, self.configuration.embedding_size])
			tf.get_variable('b_bert', [1, self.configuration.embedding_size])
			
class RecursiveModelWithParamsForDepths(RecursiveModel):
	def __init__(self, configuration):
		super().__init__(configuration)
	
	@overrides(RecursiveModel)
	def get_tensor(self, node):			
		if node.is_leaf == True:
			with tf.variable_scope('bert_layer', reuse=True):
				W_bert = tf.get_variable('W_bert', trainable=True)
				b_bert = tf.get_variable('b_bert', trainable=True)
			node_tensor = tf.nn.relu(tf.matmul(node.bert_embedding, W_bert) + b_bert)
		else:	
			with tf.variable_scope(f'layer_depth{node.depth}', reuse=True):
				W = tf.get_variable(f'W_d{node.depth}', trainable=True)
				b = tf.get_variable(f'b_d{node.depth}', trainable=True)
							
			children_tensors = [self.get_tensor(child) for child in node.children]
			mean_children_tensors = tf.reduce_mean(children_tensors, axis=0)
			node_tensor = tf.nn.relu(tf.add(tf.matmul(mean_children_tensors, W), b))
			
		node_tensor = tf.nn.dropout(node_tensor, rate=1.0 - self.configuration.keep_prob)									
		return node_tensor
	
	@overrides(RecursiveModel)
	def _compute_loss(self, label, logit):
		cross_entropy_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=tf.constant(label)))
		
		Ws = []
		with tf.variable_scope('output_layer', reuse=True):
			W = tf.get_variable('W0', trainable=True)
			Ws.append(W)

		for d in range(0, self.configuration.max_depth):
			with tf.variable_scope(f'layer_depth{d}', reuse=True):
				W = tf.get_variable(f'W_d{d}', trainable=True)
				Ws.append(W)
			
		with tf.variable_scope('bert_layer', reuse=True):
			W = tf.get_variable('W_bert', trainable=True)
			Ws.append(W)

		total_weight_loss = tf.reduce_sum([tf.nn.l2_loss(W) for W in Ws])
		total_loss = self.configuration.l2 * total_weight_loss + cross_entropy_loss
		return total_loss
	
	@overrides(RecursiveModel)
	def get_variables(self):
		with tf.variable_scope('output_layer', reuse=False):
			tf.get_variable('W0', [self.configuration.embedding_size, self.configuration.n_classes], trainable=True)
			tf.get_variable('b0', [1, self.configuration.n_classes], trainable=True)
			
		for d in range(0, self.configuration.max_depth):
			with tf.variable_scope(f'layer_depth{d}', reuse=False):
				tf.get_variable(f'W_d{d}', [self.configuration.embedding_size, self.configuration.embedding_size], trainable=True)
				tf.get_variable(f'b_d{d}', [1, self.configuration.embedding_size], trainable=True)

		with tf.variable_scope('bert_layer', reuse=False):
			tf.get_variable('W_bert', [self.configuration.bert_embedding_size, self.configuration.embedding_size], trainable=True)
			tf.get_variable('b_bert', [1, self.configuration.embedding_size], trainable=True)


class OrderedTree():
	def __init__(self, tree, embed_size):
		self.ordered_nodes = []
		self.__flatten_tree__(tree.root)
		self.isleaf = [node.is_leaf for node in self.ordered_nodes]
		self.bert_embeddings = [node.bert_embedding if node.is_leaf else [np.zeros(embed_size).reshape(-1)] for node in self.ordered_nodes]
		self.parentindex = [-1 if node.parent is None else self.ordered_nodes.index(node.parent) for node in self.ordered_nodes]
		self.depth = [node.depth for node in self.ordered_nodes]
		self.label = tree.label
		
	def __flatten_tree__(self, node):
		for child in node.children:
			self.__flatten_tree__(child)
		self.ordered_nodes.append(node)	
	
class SemiDynamicRecursiveModel(RecursiveModel):
	
	
	def __init__(self, configuration):
		
		#self.embedding_size = 3
		#self.bert_embedding_size = 3
		#self.n_classes=10
		#self.max_depth = 4
		
		super().__init__(configuration)
		self.configuration = configuration
		
		#~ print(len(self.train_trees))
		
		if self.configuration.verbose > 0:
			sys.stdout.write('Flatten Training Trees:\n')
		self.train_trees = [OrderedTree(tree, self.configuration.bert_embedding_size) for tree in (tqdm(self.train_trees) if self.configuration.verbose > 0  else self.train_trees)]
		print('Before', len(self.val_trees))
		if self.val_trees is not None:
			if self.configuration.verbose > 0:
				sys.stdout.write('Flatten Validation Trees:\n')		
			self.val_trees = [OrderedTree(tree, self.configuration.bert_embedding_size) for tree in (tqdm(self.val_trees) if self.configuration.verbose > 0 else self.val_trees)]
		print('After', len(self.val_trees))
		if self.configuration.verbose > 0:
			sys.stdout.write('Flatten Testset Trees:\n')		
		self.test_trees = [OrderedTree(tree, self.configuration.bert_embedding_size) for tree in (tqdm(self.test_trees) if self.configuration.verbose > 0 else self.test_trees)]
		
		tf.disable_eager_execution()
		
		#~ self.bert_embeddings = [[[2.0, 3.0, 4.0]], [[1.9, 0.0, 1.0]], [[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]], [[4.0, 2.0, 1.0]], [[4.0, 2.0, 1.0]], [[1.0, 1.0, 1.0]], [[4.0, 2.0, 1.0]], [[4.0, 2.0, 1.0]]]
		#self.parentindex = [3, 3, 3, 5, 5, -1]
		#self.is_leaf = [True, True, True, False, True, False]
		
		# a boolean array to determine if a node is leaf or not
		self.isleaf_placeholder = tf.placeholder(tf.bool, (None), name='isleaf_placeholder')
		# an int array for bert embeddings of leaves		
		self.bertembedd_placeholder = tf.placeholder(tf.float32, (None), name='bertembedd_placeholder')
		# an int array to retrieve the parent of the node
		self.parentindex_placeholder = tf.placeholder(tf.int32, (None), name='parentindex_placeholder')
		self.label_placeholder = tf.placeholder(tf.int32, (None), name='label_placeholder')
		self.depth_placeholder = tf.placeholder(tf.int32, (None), name='depth_placeholder')
		
		Ws = []
		with tf.variable_scope('bert_layer'):
			W_bert = tf.get_variable('W_bert', [self.configuration.bert_embedding_size, self.configuration.embedding_size], trainable=True)
			b_bert = tf.get_variable('b_bert', [1, self.configuration.embedding_size], trainable=True)
			Ws.append(W_bert)
			
		weights_arr = []
		bias_arr = []
		with tf.variable_scope(f'internal_layer'):
			for d in range(1, self.configuration.max_depth):
				W_d = tf.get_variable(f'W_d{d}', [self.configuration.embedding_size, self.configuration.embedding_size], trainable=True)
				b_d = tf.get_variable(f'b_d{d}', [1, self.configuration.embedding_size], trainable=True)
				Ws.append(W_d)
				weights_arr.append(W_d)
				bias_arr.append(b_d)
		weights_arr = tf.convert_to_tensor(weights_arr)
		bias_arr = tf.convert_to_tensor(bias_arr)
		
		#~ with tf.variable_scope(f'internal_layer'):
			#~ W = tf.get_variable(f'W', [self.configuration.embedding_size, self.configuration.embedding_size], trainable=True)
			#~ b = tf.get_variable(f'b', [1, self.configuration.embedding_size], trainable=True)
			#~ Ws.append(W)

		with tf.variable_scope('dense_layer'):
			W1 = tf.get_variable('W_dense1', [self.configuration.embedding_size, self.configuration.embedding_size], trainable=True)
			W2 = tf.get_variable('W_dense2', [self.configuration.embedding_size, self.configuration.embedding_size], trainable=True)
			b1 = tf.get_variable('b_dense1', [1, self.configuration.embedding_size], trainable=True)
			b2 = tf.get_variable('b_dense2', [1, self.configuration.embedding_size], trainable=True)
			Ws.append(W1)
			Ws.append(W2)			
		
		
		with tf.variable_scope('output_layer'):
			W_out = tf.get_variable('W_out', [self.configuration.embedding_size, self.configuration.n_classes], trainable=True)
			b_out = tf.get_variable('b_out', [1, self.configuration.n_classes], trainable=True)
			Ws.append(W_out)
			
						
		def get_leaf_embedding(node_index):	
			node_tensor = tf.gather(self.bertembedd_placeholder, node_index)
			return tf.nn.relu(tf.add(tf.matmul(node_tensor, W_bert), b_bert))
			
		def get_internal_embedding(tensor_arr, node_index):	
			children_indices = tf.squeeze(tf.where(tf.equal(self.parentindex_placeholder, node_index)))	
			all_tensors = tensor_arr.stack()
			depth = tf.gather(self.depth_placeholder, node_index)
			W = tf.gather(weights_arr, depth)
			b = tf.gather(bias_arr, depth)
			children_tensors = tf.gather(all_tensors, children_indices)
			mean_children = tf.reduce_mean(children_tensors, axis=0)
			return tf.nn.relu(tf.add(tf.matmul(mean_children, W), b))
			#n_children = tf.shape(children_tensors)[0]
			#children_tensors = tf.reshape(children_tensors, [1, -1])
			#W_aux = tf.slice(W, [0, 0], [n_children * self.configuration.embedding_size,-1])	
			#return tf.nn.relu(tf.add(tf.matmul(children_tensors, W_aux), b))
		
		def body_subtree(tensor_arr, node_index):
			isleaf = tf.gather(self.isleaf_placeholder, node_index)
			node_tensor = tf.cond(isleaf, lambda: get_leaf_embedding(node_index), lambda: get_internal_embedding(tensor_arr, node_index))
			node_tensor = tf.nn.dropout(node_tensor, rate=1.0 - self.configuration.keep_prob)									
			tensor_arr = tensor_arr.write(node_index, node_tensor) # push back the tensor to the end of TensorArray
			node_index = tf.add(node_index, 1)
			return tensor_arr, node_index		
		
		tensor_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False, infer_shape=False)
		N = tf.shape(self.parentindex_placeholder)[0]
		condition = lambda tensor_arr, idx: tf.less(idx, N)
		self.tensor_arr, _ = tf.while_loop(condition, body_subtree, [tensor_arr, 0], parallel_iterations=1)
		root_tensor = self.tensor_arr.read(self.tensor_arr.size() - 1)
			
		z = tf.nn.relu(tf.add(tf.matmul(root_tensor, W1), b1))
		z = tf.nn.dropout(z, rate=1.0 - self.configuration.keep_prob)									
		z = tf.nn.relu(tf.add(tf.matmul(z, W2), b2))
		z = tf.nn.dropout(z, rate=1.0 - self.configuration.keep_prob)
		logits = tf.add(tf.matmul(z, W_out), b_out)										
		#~ logits = tf.add(tf.matmul(root_tensor, W_out), b_out)
		
		def compute_loss(logits):
			cross_entropy_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.label_placeholder))
			#~ return cross_entropy_loss # REMOVE L2 REGULATIONS
			total_weight_loss = tf.reduce_sum([tf.nn.l2_loss(W) for W in Ws])
			loss = self.configuration.l2 * total_weight_loss + cross_entropy_loss
			return loss		
		
		self.loss = compute_loss(logits)
		#self.optimizer = tf.train.GradientDescentOptimizer(self.configuration.lr).minimize(self.loss)
		self.optimizer = tf.train.AdamOptimizer(3e-5).minimize(self.loss)
		self.prediction = tf.squeeze(tf.argmax(logits, 1))

		#X = self.tensor_arr.stack()
		#X = self.__get_leaf_embedding__(2)

		#~ with tf.Session() as sess:
			#~ sess.run(tf.compat.v1.global_variables_initializer())

				#~ #print('Here!')
				#~ x, n = sess.run([logit, N], feed_dict=feed_dict)
				#~ #print(x.shape)
				#~ #print('>' * 10)
				#~ #for i, xx in enumerate(x):
				#~ #	print(i, xx.shape)
				#~ #print('*' * 500)

	def get_feed_dict(self, tree):
		return {
			 self.isleaf_placeholder: tree.isleaf,
			 self.parentindex_placeholder: tree.parentindex,
			 self.bertembedd_placeholder: tree.bert_embeddings,
			 self.depth_placeholder: tree.depth,
			 self.label_placeholder: [tree.label]
		}
		
	@overrides(RecursiveModel)	
	def run_epoch(self, epoch_number):
		random.shuffle(self.train_trees)
		losses = []
		if not os.path.exists('./weights'):
			os.makedirs('./weights')
		with tf.Session() as sess:
			if epoch_number == 0:
				sess.run(tf.global_variables_initializer())
			else:
				saver = tf.train.Saver()
				saver.restore(sess, self.weights_path)
				
			for pos, tree in enumerate(self.train_trees):
				# TODO: enable to receive a batch of trees to be fed into the model
				l, _ = sess.run([self.loss, self.optimizer], feed_dict=self.get_feed_dict(tree))
				losses.append(l)
					
				if self.configuration.verbose > 0:
					sys.stdout.write(f'\r Epoch:{epoch_number + 1}/{self.configuration.n_epochs}, {pos + 1}/{len(self.train_trees)}: loss={np.mean(losses)}')
					sys.stdout.flush()
					
			saver = tf.train.Saver()
			saver.save(sess, self.weights_path)
		
		if self.configuration.verbose > 0:
			sys.stdout.write('\n')
			sys.stdout.flush()		
		
		train_predictions, train_losses = self.predict(self.train_trees, weights_path=self.weights_path)
		if self.val_trees is None:
			val_predictions, val_losses = None, None
		else:
			val_predictions, val_losses = self.predict(self.val_trees, weights_path=self.weights_path)
		
		train_labels = [t.label for t in self.train_trees]
		val_labels = None if self.val_trees is None else [t.label for t in self.val_trees]
		train_accuracies = np.equal(train_labels, train_predictions)
		val_accuracies = None if self.val_trees is None else np.equal(val_labels, val_predictions)
		
		return train_accuracies, val_accuracies, train_losses, val_losses
		
	@overrides(RecursiveModel)
	def predict(self, trees, weights_path):
		predictions, losses = [], []
		pos = 0
		if self.configuration.verbose > 0:
			sys.stdout.write(' Prediction:\n')
			sys.stdout.flush()
		with tf.Session() as sess:			
			saver = tf.train.Saver()
			saver.restore(sess, weights_path)
			for tree in (tqdm(trees) if self.configuration.verbose > 0 else trees):
				y_pred, l = sess.run([self.prediction, self.loss], feed_dict=self.get_feed_dict(tree))
				predictions.append(y_pred)
				losses.append(l)
		return predictions, losses
	

def test_model1(bert_folder_path, split_size_ratio, val_split_ratio=0.1):
	input_df_train, input_df_test = utils.get_train_test(input_df_address, split_size_ratio=split_size_ratio)
	n_classes = len(set(input_df_train['class'].tolist() + input_df_test['class'].tolist()))	
	configuration = Configuration(input_df_train, input_df_test, max_depth=4, val_split_ratio=val_split_ratio, n_classes=n_classes, bert_folder_path=bert_folder_path) 
	model = RecursiveModel(configuration)
	start_time = time.time()
	model.train_model()
	print(f' Elapsed time (training of {type(model).__name__}) = {round(time.time() - start_time, 2)}s')
	model.evaluate_testset()
	
def test_model2(input_df_address, bert_folder_path, split_size_ratio, val_split_ratio=0.1):
	input_df_train, input_df_test = utils.get_train_test(input_df_address, split_size_ratio=split_size_ratio)
	n_classes = len(set(input_df_train['class'].tolist() + input_df_test['class'].tolist()))	
	configuration = Configuration(input_df_train, input_df_test, val_split_ratio=val_split_ratio, n_classes=n_classes, bert_folder_path=bert_folder_path) 
	model = RecursiveModelWithLessParams(configuration)
	start_time = time.time()
	model.train_model()
	print(f' Elapsed time (training of {type(model).__name__}) = {round(time.time() - start_time, 2)}s')
	model.evaluate_testset()
	
def test_model3(input_df_address, bert_folder_path, split_size_ratio, val_split_ratio=0.1):
	input_df_train, input_df_test = utils.get_train_test(input_df_address, split_size_ratio=split_size_ratio)
	n_classes = len(set(input_df_train['class'].tolist() + input_df_test['class'].tolist()))	
	configuration = Configuration(input_df_train, input_df_test, max_depth=4, val_split_ratio=val_split_ratio, n_classes=n_classes, bert_folder_path=bert_folder_path) 
	model = RecursiveModelWithParamsForDepths(configuration)	
	start_time = time.time()
	model.train_model()
	print(f' Elapsed time (training of {type(model).__name__}) = {round(time.time() - start_time, 2)}s')
	model.evaluate_testset()
	
def semi_dynamic_model(input_df_address, bert_folder_path, split_size_ratio, val_split_ratio=0.1):
	input_df_train, input_df_test = utils.get_train_test(input_df_address, split_size_ratio=split_size_ratio)
	
	#input_df_train = input_df_train.iloc[8]
	#input_df_train = input_df_train[:20]
	#input_df_test = input_df_test[:20]	
	
	n_classes = len(set(input_df_train['class'].tolist() + input_df_test['class'].tolist()))	
	configuration = Configuration(input_df_train, input_df_test, max_depth=4, n_epochs=100, val_split_ratio=val_split_ratio, n_classes=n_classes, bert_folder_path=bert_folder_path) 
	model = SemiDynamicRecursiveModel(configuration)	
	start_time = time.time()
	model.train_model()
	print(f' Elapsed time (training of {type(model).__name__}) = {round(time.time() - start_time, 2)}s')
	model.evaluate_testset()

if __name__ == '__main__':	
	bert_folder_path=os.path.join('/home/vahidsanei_google_com/', 'data','uncased_L-12_H-768_A-12')
	input_df_address = '/home/vahidsanei_google_com/data/yelp_data/category/df_with_trees_v0.csv'

	semi_dynamic_model(input_df_address, bert_folder_path, split_size_ratio=0.8)
