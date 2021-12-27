from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import torch.nn.functional as F
import torchmetrics
from preprocess import LoadJson, Preprocess
from model import NNetwork
import torch
from torch import nn

def train(model: nn.Module, dataloader, valloader, optimizer, loss_senti, loss_asp, device, epoch=1):

  """
    Arguments
    ---------
    model : the training model
    dataloader: the training dataset loader
    valloader: validation dataset loader
    optimizer: optimizer of your choice
    loss_senti: training loss for sentiment classification
    loss_asp: training loss for aspect classification
    epoch: number of iterations
    
    """

  for j in range(epoch):
    model.train()
    for i,(x,y,sent,asp) in enumerate(dataloader):
      x=x.to(device)
      y = torch.tensor(y).to(device)
      sent = torch.tensor(sent,dtype=torch.float).to(device)
      asp = torch.tensor(asp).to(device)
      #y=y.to(device,torch.long)
      pred_senti, pred_asp = model(x,y)
      error_asp  = loss_asp(pred_asp, asp)
      error_sent = loss_senti(pred_senti.view(len(pred_senti)), sent)
      optimizer.zero_grad()
      error_asp.backward(retain_graph=True)
      error_sent.backward(error_sent)
      optimizer.step()
      acc = metric(pred_senti.view(len(pred_senti)), sent.int())
      f1_met = f1(torch.argmax(pred_asp.squeeze(), dim=1), asp)
    print("Epoch - {} / {}".format(j+1,epoch))
    print('Total Loss - {}, Accu - {}, F1 - {}'.format(error_sent.item() + error_asp.item(), acc, f1_met))
    val(model, valloader, loss_senti, loss_asp, device)

def val(model: nn.Module, dataloader, loss_senti, loss_asp, device):

  """
    Arguments
    ---------
    model : the validating model
    dataloader: the validating dataset loader
    loss_senti: training loss for sentiment classification
    loss_asp: training loss for aspect classification
    
    """

  model.eval()
  for i,(x,y,sent,asp) in enumerate(dataloader):
      x=x.to(device)
      y = torch.tensor(y).to(device)
      sent = torch.tensor(sent,dtype=torch.float).to(device)
      asp = torch.tensor(asp).to(device)
      #y=y.to(device,torch.long)
      pred_senti, pred_asp = model(x,y)
      error_asp  = loss_asp(pred_asp, asp)
      error_sent = loss_senti(pred_senti.view(len(pred_senti)), sent)
      acc = metric(pred_senti.view(len(pred_senti)), sent.int())
      f1_met = f1(torch.argmax(pred_asp.squeeze(), dim=1), asp)
      print('Total Val Loss - {}, Accu - {}, F1 - {}'.format(error_sent.item() + error_asp.item(), acc, f1_met))

if __name__ == '__main__':
  

  #loading the training, validation and testing dataset from json file
  dataset = LoadJson('/content/jack/data/sentihood/sentihood-train.json')
  val_data = LoadJson('/content/jack/data/sentihood/sentihood-dev.json')
  test_data = LoadJson('/content/jack/data/sentihood/sentihood-test.json')

  prep = Preprocess()


  #preparing separate training vocabulary of text and aspect unique words
  vocab_text = build_vocab_from_iterator(prep.yield_tokens(dataset, False), specials=["<unk>", "<pad>"], special_first=True)
  vocab_text.set_default_index(vocab_text["<unk>"])

  vocab_asp = build_vocab_from_iterator(prep.yield_tokens(dataset, True), specials=["<unk>"], special_first=True)
  vocab_asp.set_default_index(vocab_asp["<unk>"])


  #preparing training, validation and testing dataloaders
  dataloader = DataLoader(dataset, batch_size=32, collate_fn=partial(prep.collate_batch, vocab_text=vocab_text, vocab_asp=vocab_asp), shuffle=False)
  val_loader = DataLoader(val_data, batch_size=len(val_data), collate_fn=partial(prep.collate_batch, vocab_text=vocab_text, vocab_asp=vocab_asp), shuffle=False)
  test_loader = DataLoader(test_data, batch_size=len(test_data), collate_fn=partial(prep.collate_batch, vocab_text=vocab_text, vocab_asp=vocab_asp), shuffle=False)

 

  #defining all necessary parameters for our model
  hidden_dim=512
  embedding_dim=512
  vocab_size = len(vocab_text)
  target_size = len(vocab_asp)
  pad_id = vocab_text(['<pad>'])[0]
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = NNetwork(hidden_dim=hidden_dim,
                   embedding_dim=embedding_dim, 
                   vocab_size=vocab_size, 
                   target_size=target_size,
                   pad_id=pad_id,
                   device = device).to(device)
  

  # defining respective loss and metrices fro aspect and sentiment classification
  loss_asp = nn.NLLLoss()
  loss_senti = nn.BCELoss()
  #optimizer = torch.optim.SGD(model.parameters(),momentum=0.9,lr=0.001)
  optimizer = torch.optim.RMSprop(model.parameters())
  metric = torchmetrics.Accuracy().to(device)
  f1 = torchmetrics.F1().to(device)



  train(model,dataloader,val_loader,optimizer,loss_senti,loss_asp,device,epoch=50)




  #testing
  reverse_asp = {v: k for k, v in vocab_asp.get_stoi().items()}
  asp_words = []
  for i,(x,y,sent,asp) in enumerate(test_loader):
        x=x.to(device)
        y = torch.tensor(y).to(device)
        sent = torch.tensor(sent,dtype=torch.float).to(device)
        asp = torch.tensor(asp).to(device)
        #y=y.to(device,torch.long)
        pred_senti, pred_asp = model(x,y)
        acc = metric(pred_senti.view(len(pred_senti)), sent.int())
        f1_met = f1(torch.argmax(pred_asp.squeeze(), dim=1), asp)
        print('Accu - {}, F1 - {}'.format(acc, f1_met))