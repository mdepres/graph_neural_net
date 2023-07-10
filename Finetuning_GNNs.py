import os
import sys
from pathlib import Path
import math
import json
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import importlib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from toolbox import losses
from toolbox.losses import coloring_loss, triplet_loss
from toolbox import metrics
from loaders.loaders import siamese_loader
from toolbox.metrics import all_losses_acc, accuracy_linear_assignment
from toolbox.utils import check_dir
from models import finetuning_models, get_siamese_model_test
from models import utils
from loaders import data_generator
from loaders.data_generator import KCOL_Generator, QAP_Generator, MBS_Generator

import wandb
wandb.login()

def get_device_config(model_path):
    """ Get the same device as used for training the pretrained model """
    config_file = os.path.join(model_path,'config.json')
    with open(config_file) as json_file:
        config_model = json.load(json_file)
    use_cuda = not config_model['cpu'] and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    return config_model, device

def compute_dataset(args,path_dataset,train=True,bs=10):
    """ Computes some examples for the task
        - args : 'data' part of the config file
        - path_dataset : path to which the dataset will be saved, to use it several times
        - train : True if this is a training dataset, False otherwise
        - bs : desired batch size"""
    num_batches = math.ceil(args['num_examples_val']/bs)
    if train:
        gene = MBS_Generator('train', args, path_dataset)
    else:
        gene = MBS_Generator('test', args, path_dataset)
    gene.load_dataset()
    loader = siamese_loader(gene, bs, gene.constant_n_vertices)
    return loader

def is_coloring(graph, coloring):
    """ Computes the number of edges incompatible in the given coloring
    This is used as a measure of accuract for the coloring problem """
    score = 0 #Number of contradictions
    n_edges = 0
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i][j]==1 :
                n_edges+=1
                if coloring[i]==coloring[j]:
                    score+=1
    return score/n_edges # Percentage of errors

def train_epoch(model, embed_model, train_loader, loss_fn, optimizer):
    """ Train the model for one epoch 
        - model : the model to be trained
        - embed_model : the pretrained model, which has a node embedder
        - train_loader : the dataset
        - loss_fn : the loss function to use for this task
        - optimizer : an initialized optimizer instance """
    model.train()
    cum_loss = 0
    for idx, (graph,tgt) in enumerate(train_loader):
        graph['input'] = graph['input'].to(device)
        
        embed = embed_model.node_embedder(graph)['ne/suffix']
        embed = torch.permute(embed,(0,2,1))
        
        inp = torch.cat((torch.permute(graph['input'],(0,2,3,1))[:,:,:,0],embed), 2)
        
        tgt = tgt['input'].type(torch.LongTensor).to(device)
        
        out = model(inp)
        out = out.view((-1,out.shape[-1])) # The dependency of nodes to graph is irrelevant now
        
        tgt = tgt.view((-1,))
        
        optimizer.zero_grad()

        loss = loss_fn(out,tgt) #loss_fn(graph['input'], out, tgt)
        loss.backward()

        optimizer.step()
        cum_loss += loss.item()
    return cum_loss / len(train_loader)


def evaluate(model, embed_model, val_loader, compute_score=True):
    """ Evaluate the model at some point in the training
        - model : the model being trained
        - embed_molde : the pretrained model whose embeddings we use
        - val_loader : the dataset
        (- compute_score : do we compute an accuracy Broken)"""
    model.eval()
    cum_loss = 0
    cum_acc = 0
    for idx, (graph, tgt) in (enumerate(valid_loader)):
        graph['input'] = graph['input'].to(device)
        embed = embed_model.node_embedder(graph)['ne/suffix']
        embed = torch.permute(embed,(0,2,1))
        
        inp = torch.cat((torch.permute(graph['input'],(0,2,3,1))[:,:,:,0],embed), 2)
        
        tgt = tgt['input'].type(torch.LongTensor).to(device)
        
        out = model(inp)
        out = out.view((-1,out.shape[-1]))
        
        tgt = tgt.view((-1,))
        
        loss = loss_fn(out, tgt)
        cum_loss += loss.item()
        
        o = out.cpu().detach().numpy()
        t = tgt.cpu().detach().numpy()
        cum_acc += accuracy_score(np.argmax(o,axis=1),t)
    print(cum_acc / len(val_loader))
    return cum_loss / len(val_loader)

def predict(model, embed_model, adj):
    """ Show the result of the model on one example """
    graph = {'input':adj.to(device)}
    embed = embed_model.node_embedder(graph)['ne/suffix']
    embed = torch.permute(embed,(0,2,1))
    
    inp = torch.cat((torch.permute(graph['input'],(0,2,3,1))[:,:,:,0],embed), 2)
    
    out = model(inp)
    return out

cwd = os.getcwd()

model_path = cwd + '/qap/expe_new/node_embedding_rec_Regular_150_0.05/05-15-23-16-14'
config_model, device = get_device_config(model_path)
load_path = model_path + '/qap_expe_new/aqljez5s/checkpoints/epoch=9-step=6250.ckpt'
embed_model = get_siamese_model_test(load_path) #config_model["data"]["test"]["path_model"])
embed_model.to(device)

config_model["data"]["test"]['connection_density'] = 0.5
config_model["data"]["test"]["num_examples_train"] = 10000
config_model["data"]["test"]["num_examples_val"] = 100
config_model["data"]["test"]["n_vertices"] = 50

args = config_model["data"]["test"]
train_loader = compute_dataset(args, cwd+'/experiments-gnn/mbs/data')
valid_loader = compute_dataset(args, cwd+'/experiments-gnn/mbs/data', train=False)

clf = SGDClassifier(loss='log_loss', warm_start=True)

for idx, (graph,tgt) in enumerate(train_loader):
    graph['input'] = graph['input'].to(device)
    embed = embed_model.node_embedder(graph)['ne/suffix']
    
    embed = embed.cpu().detach().numpy()
    embed = np.swapaxes(embed,1,2)
    embed = np.resize(embed, (embed.shape[0]*embed.shape[1],embed.shape[-1]))
    embed = np.hstack((graph['input'].cpu().detach().numpy(),embed))
    #embed = np.hstack((embed,np.random.random((embed.shape[0],12))*np.max(embed))) 
    print(embed.shape)
    tgt = tgt['input']
    tgt = np.resize(tgt, (tgt.shape[0]*tgt.shape[1],))
    
    clf.partial_fit(embed, tgt, classes=np.arange(2))
    
acc = 0
edge_score = 0

for idx, (graph,tgt) in enumerate(valid_loader):
    graph['input'] = graph['input'].to(device)
    embed = embed_model.node_embedder(graph)['ne/suffix']
    
    embed = embed.cpu().detach().numpy()
    embed = np.swapaxes(embed,1,2)    
    embed = np.resize(embed, (embed.shape[0]*embed.shape[1],embed.shape[-1]))
    embed = np.hstack((graph['input'].cpu().detach().numpy(),embed))
    #embed = np.hstack((embed,np.random.random((embed.shape[0],12))*np.max(embed))) 
    tgt = tgt['input']
    tgt = np.resize(tgt, (tgt.shape[0]*tgt.shape[1],))
    
    pred = clf.predict(embed)
    for e in range(graph['input'].shape[0]): #For each graph in the batch
        adj = graph['input'][e]
        n_v = config_model["data"]["test"]["n_vertices"]
        edge_score += is_coloring(adj[0],pred[e*n_v:(e+1)*n_v])
    acc+=accuracy_score(pred,tgt)

print(acc/idx, edge_score/idx)
