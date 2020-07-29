import os
import numpy as np

def word2vec(dim=100):
  try:
    file_path = os.path.join('..', '..', 'glove_data', 'glove.6B.{}d.txt'.format(dim))
  except FileNotFoundError:
    print('File not found')
    return None
  res = {}
  with open(file_path) as file:
    for s in file:
      arr = s.split()
      word = arr[0]
      embedding = np.asarray(arr[1:], dtype='float32')
      res[word] = embedding
  return res
