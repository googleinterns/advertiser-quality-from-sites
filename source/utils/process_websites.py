import sys
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
from langdetect import detect_langs
from tqdm import tqdm
import nltk
from nltk import tokenize

def visible_tags(item):
    return not item.parent.name in {'meta', 'head', 'script', 'style', '[document]'} and not isinstance(item, Comment)
    
def get_sentences(texts):
	texts = tokenize.sent_tokenize(texts)
	res = []
	for text in texts:
		text = text.replace('`', '')
		text = text.replace('\\n', ' ').replace('\\r', '').replace('\\t', '')
		text = re.sub('[^a-zA-Z0-9\s]', '', text)
		text = text.split()
		text = ' '.join(text)
		res.append(text)
	return res

def get_corpus(df, output_directory=None, df_address_with_corpus=None, sentence_split=False):
    if sentence_split == True:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            if sys.version_info > (3.0):
                os.system('python3 -m nltk.downloader stopwords')
            else:
                os.system('pyhton -m nltk.downloader.stopwords')
            
    if not df_address_with_corpus is None:
        print('The dataframe already exists. We load the existing file ...')
        df = pd.read_csv(df_address_with_corpus)
    else:
        if os.path.isdir(output_directory) is False:
            print('Error: the output path does not exists!')
            return
        is_eng = []
        webpage_corpus = []
        max_text_size = 20 # maximum size for language detection
        for page_content in tqdm(df['webpage_text']):
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
            if sentence_split == True:
                visible_texts = get_sentences(visible_texts)
                visible_texts = '#'.join(visible_texts)
            else:
                visible_texts = visible_texts.replace('`', '')
                visible_texts = visible_texts.replace('\\n', ' ').replace('\\r', '').replace('\\t', ' ')
                #visible_texts = re.sub('\\?', '', visible_texts)
                visible_texts = re.sub('[^a-zA-Z0-9\s]', '', visible_texts)
                visible_texts = visible_texts.split()
                visible_texts = ' '.join(visible_texts)
            try:
                langs = detect_langs(visible_texts[:max_text_size])
                for i in range(min(2, len(langs))):
                    if langs[i].lang == 'en':
                        is_eng[-1] = True
                        webpage_corpus[-1] = visible_texts
            except Exception as e:
                #print(e)
                pass

        df['is_eng'] = is_eng
        df['webpage_corpus'] = webpage_corpus
        try:
            output_directory = '' if output_directory is None else output_directory
            file_name = 'business_with_corpus.csv' if sentence_split == False else 'business_with_corpus_sent_split.csv'
            df.to_csv(os.path.join(output_directory, file_name))
        except Exception as e:
            print(e)
    return df

def clean_categories(df_in, map_classes):
    df = df_in[df_in['categories'].notnull()]
    df['categories'] = df['categories'].apply(lambda x: re.split('[,;&]', x))
    cat = {}
    bad = []
    for x in df['categories']:
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
            
    new_cat = []
    val = 0
    for arr in df['categories']:
        new_cat.append(None)
        for x in arr:
            cls = x.strip()
            if not cls in map_classes:
                continue
            val += 1
            new_cat[-1] = map_classes[cls]
            break
            
    df['categories'] = new_cat
    df = df[df['categories'].notnull()]
    return df
    
if __name__ == '__main__':
	file_ = '/home/vahidsanei_google_com/data/yelp_data/updated_large/business.csv'
	df = pd.read_csv(file_)
	saved = df
	
	#df = saved
	#get_corpus(df, output_directory='/home/vahidsanei_google_com/data/yelp_data/updated_large/')
	
	df = saved
	get_corpus(df, output_directory='/home/vahidsanei_google_com/data/yelp_data/updated_large/', sentence_split=True)
