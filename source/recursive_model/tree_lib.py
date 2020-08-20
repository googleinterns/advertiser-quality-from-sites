import sys, os, re
sys.path.append(os.getcwd() + '/..')

import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from bs4.element import Comment, Tag, NavigableString

from utils import utils

class Tree:
	def __init__(self, encoded_tree):
		if encoded_tree is None:
			self.label = None
			self.root = Node() # word count will be zero
		else:
			idx = encoded_tree.rfind('->')
			label_part = encoded_tree[idx + 2:]
			self.label = float(label_part)
			encoded_tree = encoded_tree[:idx]
			self.root = self.tree_constructor(encoded_tree)
	
	def tree_constructor(self, encoded_tree, depth=0, parent=None):
		encoded_tree = encoded_tree[1:-1]
		node = Node()
		node.depth = depth
		node.parent = parent
		
		if '(' not in encoded_tree:
			node.content = encoded_tree
			node.wordcount = len(encoded_tree.split())
			node.is_leaf = True
			return node
		
		beg_pos, balance = 0, 0
		segments = []
		for pos in range(0, len(encoded_tree)):
			if encoded_tree[pos] == '(': balance += 1
			elif encoded_tree[pos] == ')': balance -= 1
			if balance == 0:
				segments.append((beg_pos, pos))
				beg_pos = pos + 1
		
		for beg, end in segments:
			child = self.tree_constructor(encoded_tree[beg: end + 1], depth=depth + 1, parent=node)
			node.children.append(child)
			node.wordcount += child.wordcount
		
		return node
		
class Node:
	def __init__(self):
		self.wordcount = 0
		self.depth = None
		self.content = None
		self.parent = None
		self.children = []
		self.is_leaf = False
		
def median_wordcounts(encoded_tree):
	def word_leaf_counts(node):
		if node.is_leaf == True:
			return node.wordcount, 1
		wordcount, leaves = 0, 0
		for child in node.children:
			wc, nl = word_leaf_counts(child)
			wordcount += wc
			leaves += nl
		return wordcount, leaves
	tree = Tree(encoded_tree)
	return word_leaf_counts(tree.root)
			
def read_trees(df, col_name='encoded_tree', verbose=1):
	if isinstance(df, pd.core.frame.DataFrame):
		if verbose > 0:
			sys.stdout.write('Reading trees:\n')
			sys.stdout.flush()
		trees = [Tree(tree_str) if col_name is not None else None for tree_str in (tqdm(df[col_name]) if verbose == 1 else df[col_name])]
		return trees
	else:
		sys.stdout.write('ERROR: input_ data is {type(input_)}, but the function admits pandas data frame type.')
		sys.stdout.flush()
		return None
	
def make_data_balanced(df, col_name='encoded_tree', min_wordcount=50, deviation_ratio_from_mncls=1.5):
	if not isinstance(df, pd.core.frame.DataFrame):
		sys.stdout.write('ERROR: input_ data is {type(input_)}, but the function admits pandas data frame type.')
		sys.stdout.flush()
		
	trees = read_trees(df, col_name)
	trees = [tree if (tree is not None) and (tree.root.wordcount >= min_wordcount) else None for tree in trees]
	count_cls = {}
	for tree in trees:
		if tree is None: continue
		if tree.label not in count_cls: count_cls[tree.label] = 0
		count_cls[tree.label] += 1
	
	mn = min(count_cls.values())
	limit = deviation_ratio_from_mncls * mn
	print(f'minimum = {mn}')
	
	count_cls = count_cls.fromkeys(count_cls, 0)
	for i in range(len(trees)):
		if trees[i] is None: continue
		if count_cls[trees[i].label] >= limit: trees[i] = None
		else: count_cls[trees[i].label] += 1
		
	df['trees'] = trees
	df = df[df['trees'].notnull()]
	df = df.drop(['trees'])
	return df
	
def get_text_from_siblings_with_common_parent(node):
	arr = []
	for sb in node.next_siblings:
		if isinstance(sb, NavigableString): continue
		if sb.parent.name == node.parent.name:
			txt = str(utils.clean_text(sb.text))
			arr.append(txt)
	return ' '.join(arr)	
		
def traverse(node, cur_depth, cur_branch, cutoff_text, max_depth):
	used_max_branches = cur_depth + 1
	children = [child for child in node.children if isinstance(child, Tag)]
	children = children[:used_max_branches]
		
	if len(children) == 0 or cur_depth == max_depth:
		leaf_ret = '('
		if node is not None:
			txt = str(utils.clean_text(node.text))
			leaf_ret += txt
			if cur_branch == used_max_branches - 1:
				txt = str(get_text_from_siblings_with_common_parent(node))
				leaf_ret = ' '.join([leaf_ret, txt])
		
		leaf_ret = ' '.join([leaf_ret, cutoff_text])		
		leaf_ret = [s.strip() for s in leaf_ret.split()]
		leaf_ret = ' '.join(leaf_ret)
		leaf_ret += ')'
		return None if len(leaf_ret) == 2 else leaf_ret

	node_ret = ''
	not_none_children = 0
	for i, child in enumerate(children):            
		passed_cutoff_text = ''
		if i == len(children) - 1:
			passed_cutoff_text = str(get_text_from_siblings_with_common_parent(child))
			passed_cutoff_text = ' '.join([passed_cutoff_text, cutoff_text])
			
		child_ret = traverse(child, cur_depth + 1, i, passed_cutoff_text, max_depth)
		if child_ret is not None:
			node_ret += child_ret
			not_none_children += 1
	
	if not_none_children == 0: return None
	elif not_none_children == 1: return node_ret
	else: return f'({node_ret})'
	
def get_depth_branches(node):
	children = [child for child in node.children if isinstance(child, Tag)]
	max_depth, max_branches = 0, len(children)
	for i, child in enumerate(children):
		d, b = get_depth_branches(child)
		max_depth = max(max_depth, d + 1)
		max_branches = max(max_branches, b)
	return max_depth, max_branches
			
def html_to_encoded_tree(html_content, max_depth, label):
	soup = BeautifulSoup(html_content, 'html.parser')
	tree_line = traverse(soup, 0, 0, '', max_depth)
	if tree_line is not None:
		tree_line = tree_line + f'->{label}\n'
	return tree_line

def get_html_depth_branches(html_content):
	soup = BeautifulSoup(html_content, 'html.parser')
	d, b = get_depth_branches(soup)
	return d, b

def add_encoded_trees_to_dataframe(df, label_colname='categories', corpus_colname='webpage_text', max_depth=4, is_multicase=True):
	df = df[(df[label_colname].notnull() & df[corpus_colname].notnull())]
	classes = sorted(list(set(df[label_colname])))
	print(classes)
	# convert labels to numbers (indices)
	if is_multicase == True:
		df['class'] = df[label_colname].apply(lambda x: classes.index(x))
	else:
		df['class'] = df[label_colname]
	df['encoded_tree'] = [html_to_encoded_tree(entry[corpus_colname], max_depth=max_depth, label=entry['class']) for _, entry in tqdm(df.iterrows())]
	return df

def test():
	df_raw = pd.read_csv('/home/vahidsanei_google_com/data/yelp_data/updated_large/business_with_corpus.csv')
	df = utils.unify_yelp_data_classes(df_raw)
	classes = list(set(df['categories']))
	df['categories'] = df['categories'].apply(lambda x: classes.index(x))
	for i, entry in tqdm(df.iterrows()):
		html = entry['webpage_text'] + '!'
		html = html[:-1]
		html_to_tree(html_content=html, label=entry['categories'], output_path_dir='/home/vahidsanei_google_com/data/yelp_data/trees/', is_new=True if i == 0 else False)
