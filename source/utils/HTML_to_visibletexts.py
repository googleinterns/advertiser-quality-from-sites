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

import sys, os, math, time, itertools, shutil, random, re, lxml
import argparse

sys.path.append(os.getcwd() + '/..')

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from bs4 import Comment
import requests
from langdetect import detect_langs
from tqdm import tqdm
import nltk
from nltk import tokenize

import utils

def visible_tags(item):
    return not item.parent.name in {'meta', 'head', 'script', 'style', '[document]'} and not isinstance(item, Comment)

def get_corpus(input_path, HTML_colname, output_directory=None):
	'''
		If the languange is not English, the website is removed. 
		Further ahead, the visible texts are cleaned.
	'''
	df = pd.read_csv(input_path)
	if os.path.isdir(output_directory) is False:
		print('Error: the output path does not exists!')
		return
		
	is_eng = []
	webpage_corpus = []
	# We determine the languange of the text 
	# by looking at its first 20 words
	max_text_size = 20
	for page_content in tqdm(df[HTML_colname]):
		is_eng.append(False)
		webpage_corpus.append(None)
		if page_content is np.nan: continue
		try:
			soup = BeautifulSoup(page_content, 'html.parser')
		except:
			continue
		texts = soup.findAll(text=True)
		visible_texts = filter(visible_tags, texts)
		visible_texts = u' '.join(s.strip() for s in visible_texts)
		if visible_texts is None: continue
		visible_texts = utils.clean_text(visible_texts)
		try:
			langs = detect_langs(visible_texts[:max_text_size])
			# determine if one the two guessed languages is English or not
			# if none of the two are English, we skip the website.
			for i in range(min(2, len(langs))):
				if langs[i].lang == 'en':
					is_eng[-1] = True
					webpage_corpus[-1] = visible_texts
		except Exception as e:
			pass

	df['is_eng'] = is_eng
	df['webpage_corpus'] = webpage_corpus
	try:
		df.to_csv(os.path.join(output_directory, 'business_with_corpus.csv'))
	except Exception as e:
		print(e)
    
if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Get Visible Texts from HTMLs'
	)
	parser.add_argument('--input_path', help='The path of the data')
	parser.add_argument('--output_directory', help='The directory of final result', default='.')
	parser.add_argument('--HTML_colname', help='The HTML column name', default='webpage_text')
	args = parser.parse_args()
	get_corpus(args.input_path, HTML_colname=args.HTML_colname, output_directory=args.output_directory)
