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

import sys, os
sys.path.append(os.getcwd() + '/..')

import source.utils

import argparse
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
	if not node.tag in POSSIBLE_TAGS: return
	tag[node] = node.tag
	# we limited the depth of DOM up to 7 for better visualizaiton
	if depth == 7: return
	c = 0
	for child in set(node.getchildren()):
		if not child.tag in POSSIBLE_TAGS: continue
		graph.add_edge(node, child)
		dfs(child, graph, tag, depth + 1)
		c += 1
		# we limited the number of branches up to 5 for better visualizaiton
		if c == 5:
			break

def generate_DOM(url):
	'''
		This function visualizes the DOM structure of the given URL link
	'''
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

	print('Number of nodes(tags) in the DOM Tree =', len(tag))
	
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

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Visualize DOM HTMLs'
	)
	parser.add_argument('--url', help='URL link of website')
	args = parser.parse_args()
	if generate_DOM(args.url) is False:
		st_args = utils.style()
		print(f'{st_args.BOLD}{st_args.RED} We couldn\'t visualize the website. Sorry :({st_args.RESET}')
