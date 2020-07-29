from get_url_links import get_saved_urls
import requests
import networkx as nx
from lxml import html
import re
from bs4 import BeautifulSoup
from bs4.element import NavigableString
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

POSSIBLE_TAGS = set(['html', 'head', 'body', 'div', 'img', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

def dfs(node, graph, tag, depth):
  if depth == 5: return
  if not node.tag in POSSIBLE_TAGS: return
  tag[node] = node.tag
  c = 0
  for child in set(node.getchildren()):
    if not child.tag in POSSIBLE_TAGS: continue
    graph.add_edge(node, child)
    dfs(child, graph, tag, depth + 1)
    c += 1
    if c == 3:
      break

def generate_DOM(url):
  try:
    req = requests.get(url, timeout=10.0, stream=True)
    html_text = req.text
  except Exception as e:
    print('Error url:', e)
    return False
  
  graph = nx.DiGraph()
  tag = {}
  html_tree = html.document_fromstring(html_text)
  
  print(type(html_tree))

  dfs(html_tree, graph, tag, 0)
  
  print('Number of nodes in the DOM Tree =', len(tag))
  #return True
  if len(tag) < 2:
    return False

  pos = graphviz_layout(graph, prog='dot')

  txt_setting = {'size': 10, 'color': 'white', 'weight': 'bold', 'horizontalalignment': 'center',
                 'verticalalignment': 'center', 'clip_on': True}
  bbox_setting = {'boxstyle': 'round, pad=0.2', 'facecolor': 'black', 'edgecolor': 'y', 'linewidth': 0}

  nx.draw_networkx_edges(graph, pos, arrows=True, arrowsize=10, width=2, edge_color='g')
  ax = plt.gca()
  
  for node, label in tag.items():
    x, y = pos[node]
    ax.text(x, y, label, bbox=bbox_setting, **txt_setting)

  ax.xaxis.set_visible(False)
  ax.yaxis.set_visible(False)
  
  plt.title(url, y=-0.01)

  title = html_tree.findtext('.//title')
  clean_title = re.sub('[^a-zA-Z0-9]', '', title)

  file_name = 'DOM_' + clean_title + '.pdf'
  plt.savefig(file_name)
  plt.show()
  print('File saved in {}'.format(file_name))

#generate first 100 DOM structure from the extracted urls
LIMIT = 5
c = 0
urls = get_saved_urls() 
for url in urls: 
  if url is not None:
    ret = generate_DOM(url)
    c += 1
    if c == LIMIT:
      break
