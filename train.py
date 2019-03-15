import os
from data.twitter_data import data
from data.twitter_data.data import SOS,EOS,UNK
import data_utils
import torch
import torch.nn as nn
from collections import Counter
import numpy as np
import random
from torch import optim
import torch.nn.functional as F
config ={}
config['twitter_datapath'] = 'data/twitter_data/'
# config['vocab_size']
config['emb_dim'] = 300
config['hid_size'] = 1024
config['num_layers'] =3
config['num_epoch'] = 200
config['batch_size'] = 32
config['teaching'] = True
config['teacher_forcing_ratio'] = 0.5
config['learning rate'] = 0.001
config['max_len'] = 20
config['model_dir']= 'save/'


metadata, idx_q, idx_a = data.load_data(config['twitter_datapath'])
SOS_token = metadata['w2idx'][SOS]
EOS_token = metadata['w2idx'][EOS]
UNK_token = metadata['w2idx'][UNK]
PAD_token = 0
config['vocab_size'] = len(metadata['idx2w'])

criterion  = nn.CrossEntropyLoss(ignore_index = EOS_token)
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 4, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")
    
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.input_size = config['vocab_size']
        self.emb_dim  = config['emb_dim']
        self.hidden_size = config['hid_size']
        self.num_layers = config['num_layers']
        self.time_step = config['max_len']
        self.embedding = nn.Embedding(self.input_size, self.emb_dim)
        self.rnn = nn.GRU(self.emb_dim, self.hidden_size,num_layers=self.num_layers,batch_first = True)
      
    def forward(self,input_data,hidden = None):
        time_step = config['max_len']
        embedded = self.embedding(input_data)
        output = embedded
        output, hidden = self.rnn(output,hidden)
        return output,hidden
    
    def init_hidden(self):
        hidden = torch.zeros(1,self.time_step , self.hidden_size, device=computing_device)
        return hidden

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.input_size = config['vocab_size']
        self.output_size = self.input_size
        self.emb_dim  = config['emb_dim']
        self.num_layers = config['num_layers']
        self.hidden_size = config['hid_size']
        self.embedding = nn.Embedding(self.input_size, self.emb_dim)
        self.rnn = nn.GRU(self.emb_dim, self.hidden_size,num_layers=self.num_layers,batch_first = True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input_data, hidden,encoder_outputs):
        batch_size = input_data.shape[1]
        time_step = input_data.shape[0]
        output = self.embedding(input_data)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output)
        return output,hidden

    def init_hidden(self,hidden):
        return hidden

def batchBLEU(output,target):
    score = 0.0
    batch_size = len(target)
    output= output.detach().cpu().numpy()
    target= target.cpu().numpy()
    for i in range(batch_size):
        score+=seqBLEU(output[i],target[i])
    return score/batch_size
    
def seqBLEU(candidate,reference):
    reference = reference.tolist()
    endidx = reference.index(EOS_token)
    reference = reference[1:endidx]
    counts = Counter(candidate)
    if not counts:
        return 0
    max_counts = {}
    reference_counts = Counter(reference)
    for ngram in counts:
        max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])
    clipped_counts = dict((ngram, min(count, max_counts[ngram])) for ngram, count in counts.items())
    return sum(clipped_counts.values()) / sum(counts.values())
    
def train():
    encoder_path= config['model_dir']+'encoder'+'.ckpt'
    decoder_path= config['model_dir']+'decoder'+'.ckpt'
    encoder = Encoder(config)
    decoder = Decoder(config)
    encoder.to(computing_device)
    decoder.to(computing_device)
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr = config['learning rate'])
    decoder_optimizer = torch.optim.Adam(decoder.parameters(),lr = config['learning rate'])
    
    min_val_loss=100
    for epoch in range(config['num_epoch']):
        _,train_loss = run_epoch(encoder,decoder,trainX,trainY,training=True,encoder_optimizer=encoder_optimizer,\
                                 decoder_optimizer = decoder_optimizer)
        bleu,val_loss = run_epoch(encoder,decoder,validX,validY,training=False,encoder_optimizer=encoder_optimizer,\
                                 decoder_optimizer = decoder_optimizer)
        if val_loss<min_val_loss:
            min_val_loss=val_loss
            torch.save(encoder.state_dict(),encoder_path)
            torch.save(decoder.state_dict(),decoder_path)
        print('Epoch %d,training loss: %.3f, validation loss:%.3f, BLEU score is:%.3f '%(epoch + 1, train_loss,val_loss,bleu))
    
    test_bleu,test_loss = run_epoch(encoder,decoder,testX,testY,training=False,encoder_optimizer=encoder_optimizer,\
                                 decoder_optimizer = decoder_optimizer)
    print('Training completed after %d epochs, BLEU score is %.3f'%(epoch+1,test_bleu))
#     print(test_bleu,test_loss)
    return

def run_epoch(encoder,decoder,feature,labels,training = False,encoder_optimizer=None,decoder_optimizer =None):
    
    batch_size = config['batch_size']
    epoch_loss = 0
    epoch_bleu = 0
    N = 1000
    N_minibatch_loss =0.0
    data_loader = data_utils.batch_generator(feature,labels,batch_size = config['batch_size'])
    
    for minibatch_count, (data, labels) in enumerate(data_loader):
        if training:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
        data,labels = torch.tensor(data,dtype=torch.long),torch.tensor(labels,dtype=torch.long)
        data,labels = data.to(computing_device),labels.to(computing_device)
        encoder_outputs, encoder_hidden = encoder(data)
        
#         print(encoder_hidden.shape)
#         assert 0==1
        decoder_hidden = encoder_hidden
        max_target_len = config['max_len']
        loss = 0
#         when training, 50% to teacher_forcing
        
        if training:
            use_teacher_forcing = True if random.random() < config['teacher_forcing_ratio'] else False
        
        #when test or valid, don't use teacher_forcing
        else: 
            use_teacher_forcing = False
        if config['teaching']:
            use_teacher_forcing = True
        
        decoder_charid = torch.zeros_like(labels)
        if use_teacher_forcing:
            decoder_charid[:,0]= torch.LongTensor([[SOS_token for _ in range(batch_size)]]).to(computing_device).reshape(-1)
            
            batch_size = labels.shape[0]
            target = labels[:,1:]
            target = target.contiguous().view(-1)
            
            decoder_input = labels[:,:-1]
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            decoder_charid[:,1:]= torch.argmax(decoder_output,dim=2)
            decoder_output = decoder_output.view(-1,config['vocab_size'])
            loss = criterion(decoder_output,target)
#             epoch_bleu += batchBLEU(decoder_charid,labels)
            if training:
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
            
        else:
            decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]]).to(computing_device).transpose(0,1)
            decoder_charid[:,0] = decoder_input.reshape(-1)
            batch_size = labels.shape[0]
            target = labels[:,1:]
            for t in range(max_target_len-1):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                output_id = torch.argmax(decoder_output.detach(),dim=2)
                decoder_charid[:,t+1]
                decoder_input = output_id
                loss += F.cross_entropy(decoder_output.view(batch_size,-1), target[:,t], ignore_index=EOS_token)
            if training:
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                
        epoch_bleu += batchBLEU(decoder_charid,labels)
        epoch_loss+=loss.detach() 
        N_minibatch_loss+=loss.detach()
#         loss = 0
#         if minibatch_count >5000:
#             print(minibatch_count)
        if (minibatch_count%N ==0) and (minibatch_count!=0):
#             print('hhahahahah',minibatch_count)
            train_flag = "Training" if training else "Validating/Testing"
            print(train_flag+' Average minibatch %d loss: %.3f'%(minibatch_count, N_minibatch_loss/N ))
            N_minibatch_loss = 0
    return epoch_bleu/minibatch_count,epoch_loss/minibatch_count
if __name__ == "__main__":
    train()