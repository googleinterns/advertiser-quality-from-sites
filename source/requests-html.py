import os
from requests_html import HTML, HTMLSession
from bs4 import BeautifulSoup as bs
import re

session = HTMLSession()

#https://www.automotivemachine.com/index.php/machine-shop

r = session.get('http://royalskincare.net/', stream=True, timeout=10.0)
r.html.render()

with open('file.txt', 'w') as f:
	f.write(r.html.html)

#soup = bs(r.text, 'lxml')

#for a in soup.find_all('img'):
import requests

req = requests.get('http://royalskincare.net/')
soup = bs(req.text, 'lxml')
print(soup.get_text())

exit(0)

r.html.render(timeout=10, sleep=10)

soup = bs(r.text, 'lxml')
print(soup.get_text())

print('*' * 100)

#images = r.html.find('.jpg')

#for img in images:
	#try:
		#print(img.text)
		##src = img.attrs['src']
		##print(src)
	#except:
		#pass





exit(0)






from requests_html import HTMLSession


session = HTMLSession()
r = session.get('https://pythonclock.org/')

r.html.render()

jstest = r.html.find('#yesnojs', first=True)
print(jstest.text)





