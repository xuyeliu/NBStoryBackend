import pickle
import re
import collections
import sys
import os
from html.parser import HTMLParser
from utils.myutils import prep, drop
import numpy as np
import networkx as nx
import statistics
from tokenizer import *
import argparse
import random
import time
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import copy
# from torch_geometric.nn.conv import MessagePassing
from models.GCNLayer_pytorch import GraphConvolution
from timeit import default_timer as timer
from utils.myutils import batch_gen, init_tf, seq2sent
# from models import CodeGNNGRU, TimeDistributed, Flatten
from utils.model import create_model
from html.parser import HTMLParser
import multiprocessing
import pickle
import networkx as nx
import re
import statistics
import numpy as np
import json
import collections
import ast
from ast2xml import ast2xml
from ast2xml import *
from utils.myutils import prep, drop

re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

def load(filename):
    return pickle.load(open(filename, 'rb'))

def save(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))

def load_good_fid():
    filename = './output/dataset.coms'
    good_fid = []
    for line in open(filename):
        tmp = [x.strip() for x in line.split(',')]
        fid = int(tmp[0])
        good_fid.append(fid)

    return good_fid

def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp)

def gen_pred(model, data, comstok, comlen, strat='greedy'):
    # right now, only greedy search is supported...
    tdats, coms, wsmlnodes, wedge_1 = data
    
    tdats = np.array(tdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wedge_1 = np.array(wedge_1)
    tdats = torch.from_numpy(tdats)
    coms = torch.from_numpy(coms)
    wsmlnodes = torch.from_numpy(wsmlnodes)
    wedge_1 = torch.from_numpy(wedge_1)
    tdats = tdats.type(torch.LongTensor)
    coms = coms.type(torch.LongTensor)
    wsmlnodes = wsmlnodes.type(torch.LongTensor)
    wedge_1 = wedge_1.type(torch.LongTensor)

    for i in range(1, comlen):
        output = model([tdats, coms, wsmlnodes, wedge_1])
        # results = model.predict([tdats, coms, wsmlnodes, wedge_1],
        #                         batch_size=batchsize)
        
        output = output.detach().numpy()
        for c, s in enumerate(output):
            coms[c][i] = np.argmax(s)
    coms = coms.detach().numpy()
    final_data = seq2sent(coms[0], comstok)

    return final_data

class MyHTMLParser(HTMLParser):
    def __init__(self):
        super(MyHTMLParser, self).__init__()
        self.parentstack = list()
        self.curtag = -1
        self.tagidx = -1
        self.graph = nx.Graph()
        self.seq = list()
        
    def handle_starttag(self, tag, attrs):
        if tag == '\n':
            return
        if tag.find('\n') != -1:
            tag = tag[:tag.find('\n')] + tag[tag.find('\n')+ 1:]
        # print(attrs)
        self.parentstack.append(self.curtag)
        self.tagidx += 1
        self.seq.append(tag)
        self.graph.add_node(self.tagidx, text=tag)
        if self.parentstack[-1] >= 0:
            self.graph.add_edge(self.parentstack[-1], self.tagidx)
        self.curtag = self.tagidx
        for data in attrs:

            if(data[1] != ''):
                if data[1].isdigit():
                    continue
                # data[1] = re_0001_.sub(re_0002, data[1])
                for d in data[1].split(' '): # each word gets its own node
                    if d != '' and d != '\n' and d != '\n\n' and d != '\n\n\n':
                        while d.find('\n') != -1:
                            d = d[:d.find('\n')] + d[d.find('\n') + 1:]
                        self.parentstack.append(self.curtag)
                        self.tagidx += 1
                        self.seq.append(d)
                        self.graph.add_node(self.tagidx, text=d)
                        self.graph.add_edge(self.parentstack[-1], self.tagidx)
                        self.curtag = self.tagidx
                        self.curtag = self.parentstack.pop()
        
    def handle_endtag(self, tag):
        if tag == '\n':
            return
        if tag.find('\n') != -1:
                        tag = tag[:tag.find['\n']] + tag[tag.find['\n']+ 1:]
        self.curtag = self.parentstack.pop()
        
    def handle_data(self, data):
        if data == '\n':
            return
        if data.find('\n') != -1:
            data = data[:data.find('\n')] + data[data.find('\n')+ 1:]
        # first, do dats text preprocessing
        data = re_0001_.sub(re_0002, data).lower().rstrip()

        # second, create a node if there is text
        if(data != ''):
            for d in data.split(' '): # each word gets its own node
                if d != '' and d != '\n':
                    if d.find('\n') != -1:
                        d = d[:d.find('\n')] + d[d.find('\n')+ 1:]
                    self.parentstack.append(self.curtag)
                    self.tagidx += 1
                    self.seq.append(d)
                    self.graph.add_node(self.tagidx, text=d)
                    self.graph.add_edge(self.parentstack[-1], self.tagidx)
                    self.curtag = self.tagidx
                    self.curtag = self.parentstack.pop()
        
    def get_graph(self):
        return(self.graph)

    def get_seq(self):
        return(self.seq)


def xmldecode(unit):
    parser = MyHTMLParser()
    parser.feed(unit)
    return(parser.get_graph(), parser.get_seq())

def w2i(word, smlstok):
    try:
        i = smlstok.w2i[word]
    except KeyError:
        i = smlstok.oov_index
    return i

def load_model():
    codetok = pickle.load(open('./final_data/code_notebook.tok', 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('./final_data/coms_notebook.tok', 'rb'), encoding='UTF-8')
    asttok = pickle.load(open('./final_data/ast_notebook.tok', 'rb'), encoding='UTF-8')
    codetok = asttok
    # seqdata = pickle.load(open('./final_data/data_notebookcdg.pkl', 'rb'))

    # allfids = list(seqdata['ctest'].keys())
    codevocabsize = codetok.vocab_size
    comvocabsize = comstok.vocab_size
    astvocabsize = asttok.vocab_size
    config = dict()
    config['codevocabsize'] = codevocabsize
    config['comvocabsize'] = comvocabsize
    config['astvocabsize'] = astvocabsize
    # print('codevocabsize {}'.format(codevocabsize))
    # print('comvocabsize {}'.format(comvocabsize))
    # print('astvocabsize {}'.format(astvocabsize))

    # set sequence lengths
    config['codelen'] = 200
    config['comlen'] = 30
    config['batch_size'] = 1
    config['maxastnodes'] = 300
    config['asthops'] = 2
    comlen = 30

    model, device = create_model(config)
    checkpoint = torch.load("./final_data/HAConvGNN_saved_model.h5", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adamax(model.parameters(), lr = 1e-3)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_func = torch.nn.CrossEntropyLoss()
    print("MODEL LOADED")
    return config, model, asttok, codetok, comstok
    
def interface(dats, config, model, smlstok, tdatstok, comstok):
    code = ' '.join(dats)
    newdats = re_0001_.sub(re_0002, code)
    xml = []
    for value in dats:
        if value.strip() == "":
            continue
        try:
            src = ast.parse(value)
            res = ast2xml().convert(src)
            xml.append(str(prettify(res)))
        except Exception as e:
            xml.append("")
            print("error on generating ast:", src)
    ## generate code sequence data
    newdats = re_0001_.sub(re_0002, code)
    tmp = newdats.split()
    if len(tmp) > 200:
        sys.exit()

    textdat = ' '.join(tmp)
    textdat = textdat.lower()
    prep('parsing xml... ')
    # lens = list()
    seqs = []
    nodes = []
    edge_1 = []
    blanks = 0
    try:
        for unit in xml:
            (graph, seq) = xmldecode(unit)
            seqs.append(seq)
            try:
                nodes.append(np.asarray([w2i(x[1]['text']) for x in list(graph.nodes.data())]))
                edge_1.append(nx.adjacency_matrix(graph))
                # print(type(nx.adjacency_matrix(graph)))
            except:
                eg = nx.Graph()
                eg.add_node(0)
                nodes.append(np.asarray([0]))
                edge_1.append(nx.adjacency_matrix(eg))
                blanks += 1
    except:
        unit = ''
        (graph, seq) = xmldecode(unit)
        nodes.append(np.asarray([w2i(x[1]['text']) for x in list(graph.nodes.data())]))
        edge_1.append(nx.adjacency_matrix(graph))
    nodes = np.array(nodes)
    comlen = 30 
    tdatlen = 200
    tdats = tdatstok.texts_to_sequences(textdat, maxlen=tdatlen)
    comstart = np.zeros(comlen)
    stk = comstok.w2i['<s>']
    comstart[0] = stk
    comment = list()
    comment.append(comstart)
    tdatseqs = list()
    comseqs = list()
    smlnodes = list()

    wedge_1 = list()

    comouts = list()

    fiddat = dict()

    newlen = 4 - len(nodes)
    if newlen < 0:
        newlen = 0
    wastnodes = nodes.tolist()
    for k in range(newlen):
        wastnodes.append(np.zeros(300, dtype='int32'))
    for i in range(0, len(wastnodes)):
        wastnodes[i] = np.array(wastnodes[i])[:300]
        tmp = np.zeros(config['maxastnodes'], dtype='int32')
        tmp[:wastnodes[i].shape[0]] = wastnodes[i]
        wastnodes[i] = np.int32(tmp)
    wastnodes = np.int32(wastnodes)
    wastnodes = np.asarray(wastnodes)
    wastnodes = wastnodes[:4,:]
    newlen = 4 - len(edge_1)
    if newlen < 0:
        newlen = 0
    for k in range(newlen):
        edge_1.append(np.zeros((300, 300), dtype='int32'))
    if newlen > 0:
        for i in range(0, 4 - newlen):
            edge_1[i] = np.asarray(edge_1[i].todense())
            edge_1[i] = edge_1[i][:config['maxastnodes'], :config['maxastnodes']]
            tmp_1 = np.zeros((config['maxastnodes'], config['maxastnodes']), dtype='int32')
            tmp_1[:edge_1[i].shape[0], :edge_1[i].shape[1]] = edge_1[i]
            edge_1[i] = np.int32(tmp_1)
    else:
        for i in range(0, len(edge_1)):
            edge_1[i] = np.asarray(edge_1[i].todense())
            edge_1[i] = edge_1[i][:config['maxastnodes'], :config['maxastnodes']]
            tmp_1 = np.zeros((config['maxastnodes'], config['maxastnodes']), dtype='int32')
            tmp_1[:edge_1[i].shape[0], :edge_1[i].shape[1]] = edge_1[i]
            edge_1[i] = np.int32(tmp_1)
    edge_1 = np.array(edge_1)
    edge_1 = edge_1[:4,:, :]

    tdatseqs = tdats[:config['codelen']]
    wedge_1.append(edge_1)
    smlnodes.append(wastnodes)
    tdatseqs = np.asarray(tdatseqs)
    smlnodes = np.asarray(smlnodes)
    wedge_1 = np.asarray(wedge_1)
    comment = np.asarray(comment)
    batch = [tdatseqs, comment, smlnodes, wedge_1]
    batch_results = gen_pred(model, batch, comstok, comlen, strat='greedy')
    return batch_results

    ## generate AST node and edges
if __name__ == "__main__":
    config, model, asttok, tdatstok, comstok = load_model()
    res = interface(["import numpy\nimport pandas", "def load_data():\n\treturn"], config, model, asttok, tdatstok, comstok)
    # print("[1 17 3 200 20 2 0 0 0 0 0 0 0]")
    print(res)