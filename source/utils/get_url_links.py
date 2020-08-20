#!/usr/bin/python
# -*- coding: utf-8 -*-
HOME_PATH = '/home/vahidsanei_google_com/'
PATH_DRIVER = '/home/vahidsanei_google_com/chromedriver/chromedriver'

import os
import datetime
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

import selenium
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

def get_urls(path):
	file = open(path, 'r')
	urls = [line.strip() for line in file.readlines()]
	return urls

def get_yelp_data(path):
	with open(path, 'r') as file:
		raw_data = file.readlines()
		raw_data = map(lambda x: x.rstrip(), raw_data)
		json_data = '[' + ','.join(raw_data) + ']'
	df = pd.read_json(json_data)
	return df


def save_content_of_websites():
	path = os.path.join(HOME_PATH, 'data/yelp_data', 'yelp_academic_dataset_business.json')
	df = get_yelp_data(path)
	n = len(df)

	urls = get_urls(os.path.join(HOME_PATH, 'data/yelp_data', 'link_list.txt'))

	print('size of url list = {}'.format(len(urls)))
	urls = [np.nan if url == 'None' or url.isdigit() else url for url in urls]
	urls = urls + list(np.nan for _ in range(n - len(urls)))

	df['url'] = urls
	df = df[pd.notnull(df['url'])]
	webpage_content = []
	c, not_found = 0, 0
	
	for url in tqdm(df['url']):
		webpage_content.append(None)
		if url == 'None':
			continue
		try:

			page = requests.get(url, stream=True, timeout=5.0)
			page.encoding = 'utf-8'
			webpage_content[-1] = page.content
		except Exception as e:
			c += 1
			
	print (len(webpage_content), c)
	
	n = len(df)
	webpage_content = webpage_content + list(np.nan for _ in range(n - len(webpage_content)))
	df['webpage_text'] = webpage_content
	df = df[pd.notnull(df['webpage_text'])]

	file_name = 'business.csv'

	folder_path = os.path.join(HOME_PATH, 'data', 'yelp_data', 'updated_test')
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	
	file_path = os.path.join(folder_path, 'business.csv')

	df.to_csv(file_path, index=False, header=True)
	print('Check out{}'.format(file_path))


	# pd.set_option('display.max_columns', None)

if __name__ == '__main__':
	save_content_of_websites()

			
