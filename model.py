import torch
from torch import nn
import torch.nn.functional as F


class NNetwork(nn.Module):
  def __init__(self, hidden_dim, embedding_dim, vocab_size, target_size, pad_id, device):
    super(NNetwork,self).__init__()

    """
    Arguments
    ---------
    hidden_sie : Size of the hidden_state of both the LSTMs
    embedding_dim : Embeddding dimension of word embeddings
    vocab_size : Size of the vocabulary containing unique words
    target_size : Size of the target (aspect) unique words
    pad_id : padding index of imput text
    
    """

    self.hidden_dim = hidden_dim
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
    self.dropout = nn.Dropout(0.2)
    self.lstm1 = nn.LSTM(embedding_dim, hidden_dim)
    self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)
    self.linear = nn.Linear(hidden_dim, target_size)
    self.sent_linear = nn.Linear(hidden_dim, 1)

  def forward(self, x, y):

    """ 
    Parameters
    ----------
    x: input_sentence of text (shape = (batch_size, ))
    y : binary vector of target entity
    
    Returns
    -------
    Two outputs one for aspect and another for sentiment
    
    """

    embeds = self.word_embeddings(x)
    lstm_out1, _ = self.lstm1(embeds)
    lstm_out1 = self.dropout(lstm_out1)
    lstm_out2, _ = self.lstm2(lstm_out1)
    lstm_out2 = self.dropout(lstm_out2)
    aspect_pre = self.linear(lstm_out2[:, -1, :]*y.view(len(y),1))
    aspect_pre = F.log_softmax(aspect_pre, dim=1)
    senti_pre = self.sent_linear(lstm_out2[:, -1, :]*y.view(len(y),1))
    senti_pre = F.sigmoid(senti_pre)
    return senti_pre, aspect_pre