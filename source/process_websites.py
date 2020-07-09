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

from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

def visible_tags(item):
    return not item.parent.name in {'meta', 'head', 'script', 'style', '[document]'} and not isinstance(item, Comment)

def get_corpus(df, output_path=None, df_address_with_corpus=None):
    if not df_address_with_corpus is None:
        print('The dataframe already exists. We load the existing file ...')
        df = pd.read_csv(df_address_with_corpus)
    else:
        if os.path.isdir(output_path) is False:
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
            visible_texts = visible_texts.replace('`', '')
            #print(visible_texts[:100])
            visible_texts = visible_texts.replace('\\n', ' ').replace('\\r', '').replace('\\t', '')
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
            output_path = '' if output_path is None else output_path
            df.to_csv(os.path.join(output_path, 'business_with_corpus.csv'))
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

def flatten_layers(root_layer):
    if isinstance(root_layer, keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer

def freeze_bert_layers(l_bert):
    """
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    """
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        elif len(layer._layers) == 0:
            layer.trainable = False
        l_bert.embeddings_layer.trainable = False


class categoryDetection:    
    def __init__(self, train, test, tokenizer: FullTokenizer, text_colname=None, label_colname=None, max_seq_len=128):
        """  
        """
        self.text_colname = 'webpage_corpus' if text_colname is None else text_colname
        if not self.text_colname in train.columns or not self.text_colname in test.columns:
            print('Error: Please specify a proper column name in the input dataframe as the corpus.')
            return
        
        self.label_colname = 'categories' if label_colname is None else label_colname
        if not self.label_colname in train.columns or not self.label_colname in test.columns:
            print('Error: Please specify a proper column name in the input dataframe as the labels.')
            return
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        if sys.version_info > (3.0):
            os.system('python3 -m nltk.downloader stopwords')
        else:
            os.system('pyhton -m nltk.downloader.stopwords')
        
        self.classes = train[self.label_colname].unique().tolist()
        self.classes.sort()
        
        train = train.dropna(subset=[self.text_colname])
        test = test.dropna(subset=[self.text_colname])
        
        self.max_seq_len = 0
        self.tokenizer = tokenizer
        (self.train_x, self.train_y), (self.test_x, self.test_y) = map(self._tokanize, [train, test])
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        self.train_x, self.test_x = map(self._cut_with_padding, [self.train_x, self.test_x])
    
    def build_model(self, bert_config_file, bert_ckpt_file, dropout=0.5, adapter_size=64):
        """
        """
        bert = self._load_bert(bert_config_file, bert_ckpt_file, adapter_size)
        input_ = keras.layers.Input(shape=(self.max_seq_len, ), dtype='int64', name="input_ids")
        x = bert(input_)
        #get the first embedding from the output of BERT
        x = keras.layers.Lambda(lambda seq: seq[:,0,:])(x)
        
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Dense(800, activation='relu')(x)
        #x = keras.layers.Dense(300, activation='relu')(x)
        output_ = keras.layers.Dense(units=len(self.classes), activation='softmax')(x)
        
        model = keras.Model(inputs=input_, outputs=output_)
        model.build(input_shape=(None, self.max_seq_len))
        
        load_stock_weights(bert, bert_ckpt_file)
        
        if adapter_size is not None:
          freeze_bert_layers(bert)
 
        return model
    
    def _load_bert(self, bert_config_file, bert_ckpt_file, adapter_size):
        try:
            with tf.io.gfile.GFile(bert_config_file, 'r') as gf:
                bert_config = StockBertConfig.from_json_string(gf.read())
                bert_params = map_stock_config_to_params(bert_config)
                bert_params.adapter_size = adapter_size
                bert = BertModelLayer.from_params(bert_params, name='bert')
                return bert
        except Exception as e:
            print(e)
    
    def _tokanize(self, df):
        """
        """
        X, y = [], []
        all_tokens = []
        for _, entry in tqdm(df.iterrows()):
            corpus, label = entry[self.text_colname], entry[self.label_colname]
            tokens = self.tokenizer.tokenize(corpus)
            tokens = self._clean_tokens(tokens)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            L = 0
            for x in tokens:
                if '#' in x: continue
                L += 1
            L -= 2
            if L <= 50:
            #    print(tokens)
                continue
            all_tokens.append(tokens)
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.max_seq_len = max(self.max_seq_len, len(ids))
            X.append(ids)
            y.append(self.classes.index(label))
            
        print('{}%'.format(len(X) / len(df) * 100.0))
        all_tokens = sorted(all_tokens, key=lambda x: len(x))
        #print(all_tokens[:20])
        
        return np.asarray(X), np.asarray(y)
    
    
    def _clean_tokens(self, tokens):
        # STOPS = set(stopwords.words('english'))
        clean_tokens = []
        for token in tokens:
            if any(map(str.isdigit, token)): 
                continue
            clean_tokens.append(token)
        return clean_tokens
    def _cut_with_padding(self, ids):
        """
        """
        X = []
        CLS_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        SEP_id = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
        for token_id in ids:
            # ignore tokens '[CLS]' and '[SEP]' for now
            arr = token_id[1:-1]
            sz = min(len(arr), self.max_seq_len - 2)
            arr = CLS_id + arr[:sz] + SEP_id
            # pad the remaining cells with zero
            arr = arr + [0] * (self.max_seq_len - len(arr))
            X.append(np.asarray(arr))
        return np.asarray(X)

    
def compile_model(cat:categoryDetection, model, validation_split=0.05, batch_size=16, n_epochs=30, shuffle=True):
    #log_dir = "/home/wliang_google_com/Documents/workspace/notebook/.log/website_rating/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
    #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')])
    print(model.summary())
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    history = model.fit(
        x=cat.train_x,
        y=cat.train_y,
        validation_split=validation_split,
        batch_size=batch_size,
        shuffle=shuffle,
        verbose=1,
        epochs=n_epochs,
        #callbacks=[tensorboard_callback],
    )
