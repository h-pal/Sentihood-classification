from string import punctuation
import spacy
import torch
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
import json

spacy_eng = spacy.load('en')
spacy_ger = spacy.load('de')

class LoadJson(IterableDataset):
  def __init__(self, name):

    """
    Arguments
    ---------
    name: name of input json file
    
    """
      self.name = name
  
  def __len__(self):
    with open(self.name) as f:
      d=json.load(f)
      b=[]
      for i in d:
        if i['opinions']:
          b.append(i)
    return len(b)

  def __iter__(self):
    with open(self.name) as f:
      d=json.load(f)
      b=[]
      for i in d:
        if i['opinions']:
          b.append(i)
      for content in b:
        yield content['text'], content['opinions'][0]['target_entity'], content['opinions'][0]['sentiment'], content['opinions'][0]['aspect']
  
  def getid(self):
    with open(self.name) as f:
      d=json.load(f)
      b=[]
      for i in d:
        if i['opinions']:
          b.append(i)
      a=[]
      for content in b:
        a.append(content['id'])
    return a

class Preprocess():

  def tokenize(self, text):
    return [token.text for token in spacy_eng.tokenizer(text)]

  def yield_tokens(self, data_iter, flag):
    if flag:
      for x,y,z,a in data_iter:
        a = a.lower().translate(str.maketrans('', '', punctuation))
        yield self.tokenize(a)
    else:
      for x,y,z,a in data_iter:
        x = x.lower().translate(str.maketrans('', '', punctuation))
        yield self.tokenize(x)

  def transform(self, text, vocab_text, vocab_asp, flag):
    if flag:
      text = text.lower().translate(str.maketrans('', '', punctuation))
      return [vocab_asp[token] for token in self.tokenize(text)]
    else:
      text = text.lower().translate(str.maketrans('', '', punctuation))
      return [vocab_text[token] for token in self.tokenize(text)]


  def collate_batch(self,batch, vocab_text, vocab_asp):
    text, entity, sentiment, aspect = [],[],[],[]
    for (x, y, z, a) in batch:
      text.append(torch.tensor(self.transform(x, vocab_text, vocab_asp, False)))
      entity.append(-1 if y == 'LOCATION1' else 1)
      sentiment.append(0 if z == 'Negative' else 1)
      aspect.append(self.transform(a, vocab_text, vocab_asp, True)[0])
    return pad_sequence(text, padding_value=vocab_text(['<pad>'])[0],batch_first=True), entity, sentiment, aspect

