import pandas as pd
import numpy as np
import gc
from tqdm.auto import tqdm
import pickle5
from copy import deepcopy
import json

from nltk import ngrams, word_tokenize, sent_tokenize
from nltk.stem.porter import *
from bs4 import BeautifulSoup
import re
import spacy
nlp = spacy.blank("en")

global input_dir, output_dir, model_name, model_type, criteria, description, test_project, data_pid, tokenizer, device

input_dir = '../input_data/'
output_dir = '../output_models_v5/'

model_name = 'roberta-large'
model_type = 'roberta_v5_0'

with open(input_dir + 'data_criteria_v5.pk5', 'rb') as f:
    criteria = pickle5.load(f)
with open(input_dir + 'data_description_v5.pk5', 'rb') as f:
    description = pickle5.load(f)
    
data_pid = np.unique(list(criteria.keys()) + list(description.keys()))

import pickle
with open('./test_label.pkl', 'rb') as f:
    test_label = pickle.load(f)
test_project = list(test_label.keys())

from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from adamp import SGDP,AdamP

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space = True)
device = torch.device('cuda')

def clean_text(tmp):
    soup = BeautifulSoup(tmp)
    text = soup.get_text(separator=" ").strip()
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\t\s*\t', ' ', text)
    text = re.sub(r'\xa0', ' ', text)
#     text = nlp(text)
    return text

class RelationData(Dataset):
    def __init__(self, select_pid):
        self.select_pid = select_pid
        self.null = [[''], []]

    def __len__(self):
        return len(self.select_pid)
    
    def mark_text(self, x):
        text = x[0]
        mark = x[1]
        st_marks = np.array([x[1] for x in mark])
        ed_marks = np.array([x[2] for x in mark])
        scores = np.array([x[-1] for x in mark])
        ed_marks = ed_marks[np.argsort(st_marks)]
        scores = scores[np.argsort(st_marks)]
        st_marks = np.sort(st_marks)

        marked_text = []
        prev_ed = 0
        for (st, ed) in zip(st_marks, ed_marks):
            marked_text += text[prev_ed:st] + [tokenizer.mask_token] + \
                            text[st:ed] + [tokenizer.mask_token]
            prev_ed = ed
        marked_text += text[prev_ed:]

        inputs = tokenizer(marked_text, padding = True, 
                           is_split_into_words = True,
                           truncation = True,
                           return_tensors = 'pt')

        n =  len(inputs['input_ids'][0])
        end_label = torch.zeros((1, n))
        start_label = torch.zeros((1, n))

        label_position = np.where(inputs['input_ids'].numpy() == tokenizer.mask_token_id)[1]

        if len(label_position) > 0:
            if len(label_position) % 2 == 1:
                label_position = label_position[:-1]
            for k, i in enumerate(np.arange(0, len(label_position), 2)):
                st = label_position[i]
                ed = label_position[i + 1]
                end_label[:, (ed - 1)] = float(scores[k] > 0.9)
                start_label[:, (st + 1)] = float(scores[k] > 0.9)

        mark_mask = inputs['input_ids'] != tokenizer.mask_token_id

        start_label = start_label[mark_mask]
        end_label = end_label[mark_mask]
        inputs = {k: v[mark_mask] for k,v in inputs.items()}
        
        return inputs, start_label, end_label

    def __getitem__(self, idx):
        pid = self.select_pid[idx]
        sample_c, sample_d = criteria.get(pid), description.get(pid)
        c_inputs, c_st, c_ed = self.mark_text(sample_c if sample_c is not None else self.null)
        d_inputs, d_st, d_ed = self.mark_text(sample_d if sample_d is not None else self.null)
        
        return [c_inputs, c_st, c_ed], [d_inputs, d_st, d_ed]
    
def padding(batch):
    lengths = [len(x[-1]) for x in batch]
    max_len = max(lengths)
    batch_input_ids, batch_attention_mask, batch_label_st, batch_label_ed = [], [], [], []

    batch_input_ids = torch.cat([torch.cat([x[0]['input_ids'], 
                       torch.LongTensor([tokenizer.pad_token_id]*(max_len - l))], 0).unsqueeze(0) for x, l in zip(batch, lengths)], 0)
    batch_attention_mask = torch.cat([torch.cat([x[0]['attention_mask'], 
                       torch.Tensor([0]*(max_len - l))], 0).unsqueeze(0) for x, l in zip(batch, lengths)], 0)
    batch_label_st = torch.cat([torch.cat([x[1], 
                       torch.LongTensor([0]*(max_len - l))], 0).unsqueeze(0) for x, l in zip(batch, lengths)], 0)
    batch_label_ed = torch.cat([torch.cat([x[2], 
                       torch.LongTensor([0]*(max_len - l))], 0).unsqueeze(0) for x, l in zip(batch, lengths)], 0)
    return {'input_ids': batch_input_ids, 'attention_mask': batch_attention_mask}, batch_label_st, batch_label_ed

def collate_function(samples):
    c_inputs, c_st, c_ed = padding([x[0] for x in samples])
    d_inputs, d_st, d_ed = padding([x[1] for x in samples])
    return [c_inputs, c_st, c_ed], [d_inputs, d_st, d_ed]


from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

class relation_model(nn.Module):
    def __init__(self, dims = 512, drop_rate = 0):
        super().__init__()
        self.LM = AutoModel.from_pretrained(model_name, 
                                           attention_probs_dropout_prob = 0,
                                           hidden_dropout_prob = 0,
                                           output_hidden_states = True)
        
        self.dims = dims
        self.section_embedding = nn.Embedding(2, self.dims)
        self.section_layer = nn.Linear(self.LM.config.hidden_size, self.dims, bias = False)
        self.summary_layer = nn.Sequential(nn.Mish(), nn.LayerNorm(self.dims))      
        
        self.summary_encoder = nn.ModuleList()
        for i in range(2):
            self.summary_encoder.append(nn.TransformerEncoderLayer(d_model = self.dims, nhead = 8,
                                                                      dim_feedforward = self.dims*4,
                                                                      dropout = drop_rate,
                                                                      batch_first = True))     
        

        self.output_layer = nn.Sequential(nn.Linear(self.dims, 2), nn.Sigmoid())
    
    def forward(self, c_in, d_in):
        hidden_c = self.LM(**c_in).last_hidden_state
        hidden_d = self.LM(**d_in).last_hidden_state
        hidden = torch.cat([hidden_c, hidden_d], 1)

        section = torch.LongTensor([0]*hidden_c.shape[1] + [1]*hidden_d.shape[1]).to(hidden.device)
        hidden = self.summary_layer(self.section_layer(hidden) + self.section_embedding(section))
        
        mask = 1 - torch.cat([c_in['attention_mask'], d_in['attention_mask']], 1)
        for i in range(2):
            hidden = self.summary_encoder[i](hidden, src_key_padding_mask = mask.bool())

        probs = self.output_layer(hidden)
        return probs
    
    
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def thres_search(y, p):
    auc = roc_auc_score(y, p)
    f1_table = []
    thres_range = np.arange(0.1, 0.9, 0.02)
    for thres in thres_range:
        f1 = f1_score(y, p > thres)
        f1_table.append([thres, f1])
    f1_table = np.array(f1_table)
    best_thres = thres_range[f1_table[:, -1].argmax(-1)]
    f1 = f1_score(y, p > best_thres)
    precision = precision_score(y, p > best_thres)
    recall = recall_score(y, p > best_thres)
    return best_thres, [auc, f1, precision ,recall]

def entity_extraction3(inputs, probs, thres):
    
    probs = probs[0][inputs['attention_mask'][0] == 1, :]
    tokens = inputs['input_ids'][0][inputs['attention_mask'][0] == 1]    
    
    entity = []
    potential_st = np.where(probs[:, 0].cpu() > thres[0])[0]
    if len(potential_st) > 0:
        for k, st in enumerate(potential_st):
            if k + 1 == len(potential_st):
                ed_probs = probs[st:, 1].cpu()
            else:
                ed_probs = probs[st:potential_st[k+1], 1].cpu()
            ed = ed_probs.argmax().item()
            if True:
                entity_token = tokens[st:(st + ed + 1)]
                keyword = tokenizer.decode(entity_token, skip_special_tokens = True).strip()
                if len(keyword) > 0:
                    entity.append(keyword)
        entity = np.unique(entity)
        entity = entity[~pd.Series([x.lower() for x in entity]).duplicated().values].tolist()
    return entity

def inference(thres_list, eval_label):
    
    model = relation_model().to(device)
    model.load_state_dict(torch.load(f'{output_dir}/{model_type}.pt'))
    model = model.eval()    

    ts_id = list(eval_label.keys())
    ts_data = RelationData(ts_id)
    ts_loader = DataLoader(ts_data, batch_size = 1, shuffle = False, collate_fn = collate_function)

    predictions = {}
    for i, (c_inputs, d_inputs) in enumerate(tqdm(ts_loader)):

        c_in = {k:v.to(device) for k,v in c_inputs[0].items()}
        c_st, c_ed = c_inputs[1].to(device), c_inputs[2].to(device)
        d_in = {k:v.to(device) for k,v in d_inputs[0].items()}
        d_st, d_ed = d_inputs[1].to(device), d_inputs[2].to(device)

        with torch.no_grad():
            probs = model(c_in, d_in)

        inputs = {k: torch.cat([c_in[k], d_in[k]], 1) for k,v in d_in.items()}
        entity_list = entity_extraction3(inputs, probs, thres = thres_list)

        predictions[ts_id[i]] = entity_list
        
    return predictions