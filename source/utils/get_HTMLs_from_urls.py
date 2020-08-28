# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import numpy as np
import json
import pandas as pd
from bs4 import BeautifulSoup
from bs4 import Comment

import lxml
import re
import requests
from requests_html import HTMLSession
from langdetect import detect_langs
from tqdm import tqdm

def get_urls(path):
	file = open(path, 'r')
	urls = [line.strip() for line in file.readlines()]
	return urls

def get_yelp_data(path):
	'''
		This function converts json file to csv
	'''
	with open(path, 'r') as file:
		raw_data = file.readlines()
		raw_data = map(lambda x: x.rstrip(), raw_data)
		json_data = '[' + ','.join(raw_data) + ']'
	df = pd.read_json(json_data)
	return df


def save_content_of_websites(input_path, urls_path, output_path):
	df = get_yelp_data(input_path)
	n = len(df)
	urls = get_urls(urls_path)

	print('size of url list = {}'.format(len(urls)))
	urls = [np.nan if url == 'None' or url.isdigit() else url for url in urls]
	urls = urls + list(np.nan for _ in range(n - len(urls)))

	df['url'] = urls
	df = df[pd.notnull(df['url'])]
	webpage_content = []
		
	for url in tqdm(df['url']):
		webpage_content.append(None)
		if url == 'None':
			continue
		try:
			page = requests.get(url, stream=True, timeout=5.0)
			page.encoding = 'utf-8'
			webpage_content[-1] = page.content
		except Exception as e:
			pass
			
	n = len(df)
	webpage_content = webpage_content + list(np.nan for _ in range(n - len(webpage_content)))
	df['webpage_text'] = webpage_content
	df = df[pd.notnull(df['webpage_text'])]

	file_name = 'business.csv'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	file_path = os.path.join(output_path, 'business.csv')

	df.to_csv(file_path, index=False, header=True)
	print(f'Check out{output_path} ...')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Get HTMLs from URL Links'
	)
	parser.add_argument('--input_path', help='The path of the data')
	parser.add_argument('--urllinks_path', help='The path of url links')
	parser.add_argument('--output_directory', help='The directory of final result', default='.')
	args = parser.parse_args()
	save_content_of_websites(args.input_path, args.urllinks_path, args.output_path)

			
