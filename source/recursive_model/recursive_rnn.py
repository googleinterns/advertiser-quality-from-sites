import sys, os, math, time, itertools, shutil, random
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

class Configuration():
	def __init__(self, input_df=None, lr=1e-2, n_epochs=100, l2=0.01, test_split_ratio=0.2, val_split_ratio=None, anneal_factor=1.5, 
			anneal_tolerance=0.98, patience_early_stopping=5, max_depth=None, n_classes=None, bert_folder_path=None, 
			bert_embedding_size=768, embedding_size=800, keep_prob=0.65, max_content_length=128, verbose=1):
		assert input_df is not None, 'input df cannot be None'
		assert n_classes is not None, '# of classes must be specified'
		assert bert_folder_path is not None, 'input files for BERT must be specified'

		self.input_df = input_df
		self.lr = lr
		self.n_epochs = n_epochs
		self.l2 = l2
		self.test_split_ratio = test_split_ratio
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
		self.weights_path = None
	
class RecursiveModel():
	def __init__(self, configuration):		
		
		self.configuration = configuration
		self.__set_model_name_and_path_weights__()				
		trees = tree_lib.read_trees(self.configuration.input_df, verbose=self.configuration.verbose)
		random.shuffle(trees)
		
		###################################
		### for testing
		#trees = trees[:100]
		###################################
			
		self.bert_model = self.load_bert()
		if self.configuration.verbose > 0:
			sys.stdout.write(' [Preprocessing step] Get embeddings of leaves contents:\n')
			sys.stdout.flush()
		for tree in (tqdm(trees) if self.configuration.verbose > 0 else trees):
			self.__content_to_embedding__(tree.root)	
					
		tf.disable_v2_behavior()
		#tf.disable_eager_execution()	

		split_size = int((1.0 - self.configuration.test_split_ratio) * len(trees))
		self.train_trees = trees[:split_size]
		self.test_trees = trees[split_size:]
		self.val_trees = None
		if self.configuration.val_split_ratio is not None:
			split_size = int((1.0 - self.configuration.val_split_ratio) * len(self.train_trees))
			self.val_trees = self.train_trees[split_size:]
			self.train_trees = self.train_trees[:split_size]
			
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
			
	def __set_model_name_and_path_weights__(self):
		self.configuration.model_name = f'{socket.gethostname()}_{self.__class__.__name__}_cls={self.configuration.n_classes}'
		self.configuration.weights_path = os.path.join('weights', self.configuration.model_name)
		
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
		output_ = keras.layers.Lambda(lambda seq: seq[:,0,:])(x)
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
		weights_path = f'weights/{self.configuration.model_name}.temp'
		while pos < len(self.train_trees):
			with tf.Graph().as_default(), tf.Session() as sess:
				self.get_variables()
				
				if epoch_number == 0 and pos == 0:
					sess.run(tf.global_variables_initializer())
				else:
					saver = tf.train.Saver()
					saver.restore(sess, weights_path)
				
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
				if not os.path.exists('./weights'):
					os.makedirs('./weights')
				saver.save(sess, weights_path)
		
		if self.configuration.verbose > 0:
			sys.stdout.write('\n')
			sys.stdout.flush()		
		
		train_predictions, train_losses = self.predict(self.train_trees, weights_path=weights_path)
		val_predictions, val_losses = self.predict(self.val_trees, weights_path=weights_path)
		
		train_labels = [t.label for t in self.train_trees]
		val_labels = [t.label for t in self.val_trees]
		
		train_accuracies = np.equal(train_labels, train_predictions)
		#print('debugging train\n', train_labels, train_predictions)
		val_accuracies = np.equal(val_labels, val_predictions)
		#print('debugging validation\n', val_labels, val_predictions)
		return train_accuracies, val_accuracies, train_losses, val_losses
			
	def train_model(self):
		train_losses = []
		train_accuracies = []
		val_losses = []
		val_accuracies = []
		last_epoch_loss = float('inf')
		best_val_loss = float('inf')
		best_val_epoch = -1
		
		for epoch in range(self.configuration.n_epochs):		
			epoch_train_accuracies, epoch_val_accuracies, epoch_train_losses, epoch_val_losses = self.run_epoch(epoch_number=epoch)
			train_acc, val_acc, train_loss, val_loss = np.mean(epoch_train_accuracies), np.mean(epoch_val_accuracies), np.mean(epoch_train_losses), np.mean(epoch_val_losses)
			sys.stdout.write(f' Train acc = {round(train_acc * 100, 3)}%, Val acc = {round(val_acc * 100, 3)}%, Train loss = {train_loss}, Val loss = {val_loss}, lr={self.configuration.lr}\n')
			sys.stdout.write(('*' * 130) + '\n')
			sys.stdout.flush()
			
			train_losses.append(np.mean(epoch_train_losses))
			train_accuracies.append(np.mean(epoch_train_accuracies))
			val_losses.append(np.mean(epoch_val_losses))
			val_accuracies.append(np.mean(epoch_val_accuracies))
			
			if train_losses[-1] > last_epoch_loss * self.configuration.anneal_tolerance:
				self.configuration.lr /= self.configuration.anneal_factor
				#sys.out.write(f'\n *** Annealed learning rate to {self.configuration.lr} *** \n')
			last_epoch_loss = train_losses[-1]
			
			if val_losses[-1] < best_val_loss:
				# TODO: for testing phase, save the current model in best_weights.temp 
				best_val_loss = val_losses[-1]
				best_val_epoch = epoch
			
			#if epoch - best_val_epoch > self.configuration.patience_early_stopping:
			#	break

class RecursiveModelWithLessParams(RecursiveModel):
	def __init__(self, configuration):
		super().__init__(configuration)
	
	# overrided method
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
	
	# overrided method
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
	
	# overrided method
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
	
	# overrided method
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
	
	# overrided method
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
	
	# overrided method
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


def test_model():
	bert_folder_path = os.path.join('/home/vahidsanei_google_com/', 'data','uncased_L-12_H-768_A-12')
	df = pd.read_csv('/home/vahidsanei_google_com/data/yelp_data/trees/df_with_trees.csv')
	#df = df[:50]
	configuration = Configuration(input_df=df, max_depth=4, val_split_ratio=0.1, test_split_ratio=0.001, n_classes=10, bert_folder_path=bert_folder_path) 
	model = RecursiveModel(configuration)
	
	start_time = time.time()
	model.train_model()
	print(f' Elapsed time (training) = {time.time() - start_time}')
	
def test_model2():
	bert_folder_path = os.path.join('/home/vahidsanei_google_com/', 'data','uncased_L-12_H-768_A-12')
	df = pd.read_csv('/home/vahidsanei_google_com/data/yelp_data/trees/df_with_trees.csv')
	#df = df[:50]
	configuration = Configuration(input_df=df, val_split_ratio=0.1, test_split_ratio=0.001, n_classes=10, bert_folder_path=bert_folder_path) 
	model = RecursiveModelWithLessParams(configuration)	

	start_time = time.time()
	model.train_model()
	print(f'Training time = {time.time() - start_time}')
	
def test_model3():
	bert_folder_path = os.path.join('/home/vahidsanei_google_com/', 'data','uncased_L-12_H-768_A-12')
	df = pd.read_csv('/home/vahidsanei_google_com/data/yelp_data/trees/df_with_trees.csv')
	#df = df[:50]
	configuration = Configuration(input_df=df, max_depth=4, val_split_ratio=0.1, test_split_ratio=0.001, n_classes=10, bert_folder_path=bert_folder_path) 
	model = RecursiveModelWithParamsForDepths(configuration)	

	start_time = time.time()
	model.train_model()
	print(f'Training time = {time.time() - start_time}')
			
if __name__ == '__main__':
	test_model2()
	
#shutil.copyfile('./weights/%s.temp.data-00000-of-00001'%self.configuration.model_name, './weights/%s.data-00000-of-00001'%self.configuration.model_name)
#shutil.copyfile('./weights/%s.temp.index'%self.configuration.model_name, './weights/%s.index'%self.configuration.model_name)
#shutil.copyfile('./weights/%s.temp.meta'%self.configuration.model_name, './weights/%s.meta'%self.configuration.model_name)
