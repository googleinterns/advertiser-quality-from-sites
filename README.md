# Intern Project: Identifying Advertiser Quality from Their Websites

Intuitively, for an ad campaign, the quality of the advertiser’s business itself can have a big impact on the campaign performance, especially conversion performance. For a restaurant, the quality can be their food, service and location that provided to the customer; for an e-commerce website, the quality can be the ease of use of their website, shipping and customer service. The quality of an advertiser can be estimated from several data sources, such as their website quality, their rating on popular rating platforms, such as Google Maps or Yelp, and/or how natively they rank on Google Search, etc. Considering estimating the quality of the advertiser does not require ads data, we would like to measure the quality of advertiser from their websites in the intern project.


Our methods to preproess the data and our neural network models are built upon Python 3, Tensorflow on CPU and GPUs. We developed two neural network models: 1) baseline model 2) DOM-based models. Both approaches use BERT-Base model with 12 layers. Further ahead, we show how to run our codes to retrieve url links, HTMLs, clearning visible texts, visualizing DOM structures, and also we show how to run our NN models to train the data and to predict the category/rating of urls links.

## Instructions

### 1. Data Preprocessing
All methods for preprocessing step are located at folder [utils](https://github.com/googleinterns/advertiser-quality-from-sites/tree/master/source/utils). Here, we show how to run the code.

**Get HTMLS of URL Links: [get_HTMLs_from_urls.py](https://github.com/googleinterns/advertiser-quality-from-sites/blob/master/source/utils/get_HTMLs_from_urls.py)** \
This code gets input path of url links which is basically a text file where every line contains a url link. It also gets a dataframe as the input that has information of businesses which are ordered the same order as the order of url links in the text file. It extracts the HTML contents of urls and store a data frame with the extractd HTMLs in the output directory address.
```python
usage: get_HTMLs_from_urls.py [-h] [--input_path INPUT_PATH]
                              [--urllinks_path URLLINKS_PATH]
                              [--output_directory OUTPUT_DIRECTORY]

Get HTMLs from URL Links

optional arguments:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        The path of the data
  --urllinks_path URLLINKS_PATH
                        The path of url links
  --output_directory OUTPUT_DIRECTORY
                        The directory of final result

```
**Get Visible Texts of HTMLS: [HTML_to_visibletexts.py](https://github.com/googleinterns/advertiser-quality-from-sites/blob/master/source/utils/HTML_to_visibletexts.py)** \
This Python code retrieves visible texts from HTMLs and then clean the text by removing non-ascii characters and replacing \t and multiple spaces with a single space. We get the input path of the data frame and the column name for HTML contents. Further, it ignores the websites that do not use English using the Python package langdetect. It saves the final dataframe in the output directory.
```python
usage: HTML_to_visibletexts.py [-h] [--input_path INPUT_PATH]
                               [--output_directory OUTPUT_DIRECTORY]
                               [--HTML_colname HTML_COLNAME]

Get Visible Texts from HTMLs

optional arguments:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        The path of the data
  --output_directory OUTPUT_DIRECTORY
                        The directory of final result
  --HTML_colname HTML_COLNAME
                        The HTML column name

```
**Visualize the DOM structure of HTML: [gen_DOM_tree.py](https://github.com/googleinterns/advertiser-quality-from-sites/blob/master/source/utils/gen_DOM_tree.py)** \
This code visualizes the DOM structure of the given HTML. For a better visualization, we limit the depth of the tree (DOM) to 7 and the maximum branches to 5. Here, you see how to run the code by passing a url link. The output pdf file is saved in the same directory as the python file.
```python
usage: gen_DOM_tree.py [-h] [--url URL]

Visualize DOM HTMLs

optional arguments:
  -h, --help  show this help message and exit
  --url URL   URL link of website
```
Here is an example:
```ruby
python gen_DOM_tree.py --url https://www.pizzahut.com
```
<p align="center">
  <img src="https://github.com/googleinterns/advertiser-quality-from-sites/blob/master/source/image_example/DOM_PizzaHutPizzaDeliveryPizzaCarryoutCouponsWingsMore.png" width = 700 height = 500/>
</p>

**Helper functions: [utils.py](https://github.com/googleinterns/advertiser-quality-from-sites/blob/master/source/utils/utils.py)** \
Here, we maintain several helper functions for cleaning texts, unifying categories, spliting the training and test sets, oversampling the training data, and plotting the distribution of the data. We listed the functions of utils below:
```python
def suppress_stdout #for supressing stdout.
def suppress_sterr #for suppressing stderr.
def unify_yelp_data_classes #unifies categories by maping a list of word to a label name.
def remove_not_loaded_websites #remove websites that are not loaded.
def oversampling # oversamples the training data.
def plot_classes_distribution #plots the distirbution of the entries for each class.
def clean_text #clean the input text by removing non-ascii characters and replace multiple spaces and \t with a single space.
def get_train_test #split the data to training and test sets.
```
**Tree Library and its helper functions: [tree_lib.py](https://github.com/googleinterns/advertiser-quality-from-sites/blob/master/source/recursive_model/tree_lib.py
)** \
This file contains several functions for converting an HTML to tree strings, a data structure for maining trees, data balancing based on the labels of trees, get the stats of trees including the number of nodes, maximum depths, and their maximum branches.

### 2. Models
**Baseline model: [baseline_model.py](https://github.com/googleinterns/advertiser-quality-from-sites/blob/master/source/baseline/baseline_model.py)** \
This code is our baseline model, which prepares the data for the model by a preprocessing step for converting texts into BERT embeddings. Then, the model starts to train and finally evaluates the test set. The other option to this model is to pass a url links and get the prediction of its category or rating. Here are the parameters for running the code:
```python
usage: baseline_model.py [-h] [--tasktype {C,R}]
                         [--input_directory INPUT_DIRECTORY]
                         [--adam_lr ADAM_LR] [--n_epochs N_EPOCHS]
                         [--val_split_ratio VAL_SPLIT_RATIO]
                         [--bert_folder_path BERT_FOLDER_PATH]
                         [--bert_embedding_size BERT_EMBEDDING_SIZE]
                         [--keep_prob KEEP_PROB]
                         [--max_content_length MAX_CONTENT_LENGTH]
                         [--n_hidden_layers N_HIDDEN_LAYERS] [--url URL]
                         [--best_weight_path BEST_WEIGHT_PATH]
                         [--chrome_path CHROME_PATH]

Baseline -- Identifying Advertiser Quality from Their Websites

optional arguments:
  -h, --help            show this help message and exit
  --tasktype {C,R}      (C) Classification or (R)Regression
  --input_directory INPUT_DIRECTORY
                        Directory for train and test data
  --adam_lr ADAM_LR     Adam learning rate
  --n_epochs N_EPOCHS   Numbrt of epochs
  --val_split_ratio VAL_SPLIT_RATIO
                        Validation size ration
  --bert_folder_path BERT_FOLDER_PATH
                        Folder path of model BERT
  --bert_embedding_size BERT_EMBEDDING_SIZE
                        BERT output embedding size
  --keep_prob KEEP_PROB
                        Kept rate of dropout layers
  --max_content_length MAX_CONTENT_LENGTH
                        Maximum content length of from each leaf of DOM
  --n_hidden_layers N_HIDDEN_LAYERS
                        Number of hidden layers
  --url URL             URL link of business website
  --best_weight_path BEST_WEIGHT_PATH
                        URL link of business website
  --chrome_path CHROME_PATH
                        The path to chrome engine for Python package selenium

```
**DOM based models: [DOMbased_model.py](https://github.com/googleinterns/advertiser-quality-from-sites/blob/master/source/recursive_model/DOMbased_model.py)** \
This file contains the implementation of DOM-based models including the Fast DOM-based model (FDBM) and their modified versions. However, when we run the code the FDBM with mean (which the children embeddings are averaged out) is run. Similar to the baseline model, we can pass a url link, otherwise (if we don't pass a url link) we can train our model. Here are the paramteres to run the code.
```python
usage: DOMbased_model.py [-h] [--tasktype {C,R}]
                         [--input_directory INPUT_DIRECTORY]
                         [--adam_lr ADAM_LR] [--n_epochs N_EPOCHS] [--l2 L2]
                         [--val_split_ratio VAL_SPLIT_RATIO]
                         [--max_depth MAX_DEPTH]
                         [--bert_folder_path BERT_FOLDER_PATH]
                         [--bert_embedding_size BERT_EMBEDDING_SIZE]
                         [--embedding_size EMBEDDING_SIZE]
                         [--keep_prob KEEP_PROB]
                         [--max_content_length MAX_CONTENT_LENGTH] [--url URL]
                         [--best_weight_path BEST_WEIGHT_PATH]
                         [--chrome_path CHROME_PATH]

Fast DOM Based Model -- Identifying Advertiser Quality from Their Websites

optional arguments:
  -h, --help            show this help message and exit
  --tasktype {C,R}      (C) Classification or (R)Regression
  --input_directory INPUT_DIRECTORY
                        Directory for train and test data
  --adam_lr ADAM_LR     Adam learning rate
  --n_epochs N_EPOCHS   Numbrt of epochs
  --l2 L2               L2 regularization factor
  --val_split_ratio VAL_SPLIT_RATIO
                        Validation size ration
  --max_depth MAX_DEPTH
                        Maximum depth for DOM based model
  --bert_folder_path BERT_FOLDER_PATH
                        Folder path of model BERT
  --bert_embedding_size BERT_EMBEDDING_SIZE
                        BERT output embedding size
  --embedding_size EMBEDDING_SIZE
                        DOM-based model output Embedding size
  --keep_prob KEEP_PROB
                        Kept rate of dropout layers
  --max_content_length MAX_CONTENT_LENGTH
                        Maximum content length of from each leaf of DOM
  --url URL             URL link of business website
  --best_weight_path BEST_WEIGHT_PATH
                        URL link of business website
  --chrome_path CHROME_PATH
                        The path to chrome engine for Python package selenium
```
## Demo
Here is a demo of FDBM for predicting the category and rating of businesses given their url links:
<p align="center">
  <img src="https://github.com/googleinterns/advertiser-quality-from-sites/blob/master/source/image_example/demo-shortversion.gif"/>
</p>

## License

Apache 2.0; see [LICENSE](LICENSE) for details.

## Disclaimer

**This is not an officially supported Google product.**


