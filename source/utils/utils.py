import sys, os, math, datetime, functools, re
sys.path.append(os.getcwd() + '/..')

import numpy as np
import pandas as pd

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
    , col_name='categories', show_skipped=False):
		
	df = df[df[col_name].notnull()]
	df[col_name] = df[col_name].apply(lambda x: re.split('[,;&]', x))
	
	if show_skipped == True:
		show_skipped_classes(df, map_classes)
		return df
		
	cat = []
	for arr in df[col_name]:
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
	df[col_name] = cat
	df = df[df[col_name].notnull()]
	if show_skipped == True:
		show_skipped_classes(df, map_classes)
	return df
	
def remove_not_loaded_websites(df, bad_keywords = {'javascript' }, col_name='webpage_corpus', min_wordcount=50):
	df = df[df[col_name].notnull()]
	for bad_keyword in bad_keywords:
		df[col_name] = df[col_name].apply(lambda x: None if bad_keyword.lower() in x.lower() else x)
		df = df[df[col_name].notnull()]
	df[col_name] = df[col_name].apply(lambda x: None if len(x.split()) < min_wordcount else x)
	df = df[df[col_name].notnull()]
	return df
	
def show_skipped_classes(df, map_classes, col_name='categories'):
	#This function is to check which entires are removed due to not belonging to any defined classes
	cat = {}
	bad = []
	for x in df[col_name]:
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
            
def plot_classes_distribution(df, col_name='categories'):
    x= df[col_name].value_counts()
    x= x.sort_index()

    plt.figure(figsize=(10,6))
    ax= sns.barplot(x.index, x.values, alpha=0.9, palette='hls')
    ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment='center', fontsize=15, rotation=80)
    plt.title(f'{col_name} distribution')
    plt.ylabel('Number of businesses', fontsize=20)
    plt.xlabel(f'{col_name}', fontsize=20)
    
    rects = ax.patches
    labels = x.values
    for rect, label in zip(rects, labels):
        ax.text(rect.get_x() + rect.get_width() * 0.5, rect.get_height() + 10, label, ha='center', va='bottom', fontsize=14)

def clean_text(text):
	res = (text + '!')[:-1]
	res = res.strip()
	res = res.replace('\'', '')
	res = re.sub(r'\\x..|\\xdd', '', res)
	res = re.sub(r'\\s|\\n|\\r|\\t|\n|\r|\t|\)|\(', ' ', res)
	res = res.replace('\x00','')
	res = res.encode("ascii", errors="ignore").decode()
	arr = res.split()
	return ' '.join(arr)
	
def get_train_test(input_df_address, split_size_ratio):
	df = pd.read_csv(input_df_address)
	split_size = int(split_size_ratio * len(df))
	input_df_train = df[:split_size]
	input_df_test = df[split_size:]
	return input_df_train, input_df_test
