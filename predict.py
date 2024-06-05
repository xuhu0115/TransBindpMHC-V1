# coding: utf-8

import models     #同级目录
import sys

import math
from sklearn import metrics
from sklearn import preprocessing
import numpy as np
import pandas as pd
import re
import time
import datetime
import random
random.seed(199901)
from scipy import interp
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from collections import OrderedDict
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import difflib
from io import BytesIO
from PIL import Image


seed = 199901
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from tqdm import tqdm

transform_benchmark = pd.read_csv("./Dataset/HLA/new_hla_pseudo.csv")

def HLA_to_pseudo(MHC_type:str):
    # 查询对应的序列
    MHC_pseudo_sequence = transform_benchmark[transform_benchmark['HLA'] == MHC_type]['sequence'].values
    #print("MHC_pseudo_sequence:",MHC_pseudo_sequence)
    return "".join(MHC_pseudo_sequence)

def predict(peptide_input, HLA_input, threshold=0.5, cut_peptide=False, cut_length=9):
    # 如果输入是文件，读取文件内容
    if peptide_input.endswith(".fasta"):
        #peptide_content = peptide_input.read().decode('utf-8')
        with open(peptide_input, 'r', encoding='utf-8') as f:
            peptide_content = f.read()
        ori_peptides = []  
        for line in peptide_content.split("\n"):
            if line.startswith(">"): 
                continue
            ori_peptides.append(line)
    elif isinstance(peptide_input, str):
        peptide_content = peptide_input
        ori_peptides = [peptide_content] 

    if HLA_input.endswith(".fasta"):
        #HLA_content = HLA_input.read().decode('utf-8')
        with open(HLA_input, 'r', encoding='utf-8') as f:
            HLA_content = f.read()
        ori_HLA_sequences = []
        ori_HLA_names = [] 
        for line in HLA_content.split("\n"):
            #print("line:",line)
            if line.startswith(">"):
                ori_HLA_name = line[1:] # 去掉">"
                ori_HLA_names.append(ori_HLA_name) 
            else:
                #sequence = HLA_to_pseudo(line)#类型转换成伪序列
                ori_HLA_sequences.append(line)
    elif isinstance(HLA_input, str):
        HLA_content = HLA_input
        ori_HLA_names = [HLA_content]
        ori_HLA_sequences = [HLA_to_pseudo(HLA_content)]

       
    print("length of ori_peptides:", len(ori_peptides))
    print("length of ori_HLA_names:", len(ori_HLA_names))
    print("length of ori_HLA_sequences:", len(ori_HLA_sequences))
    print("ori_peptides:",ori_peptides)
    print("ori_HLA_names:", ori_HLA_names)
    print("ori_HLA_sequences:",ori_HLA_sequences)

    # 检查长度一致  
    if len(ori_peptides) != len(ori_HLA_sequences):
        print("Number not match! Please ensure the same number of HLAs and peptides.") 
        return "" # 返回空不退出程序


    peptides, HLA_names, HLA_sequences = [], [], []
    for pep, hla_name, hla_seq in tqdm(zip(ori_peptides, ori_HLA_names, ori_HLA_sequences),total=len(ori_peptides), desc='Predicting'):
        if not (pep.isalpha() and hla_seq.isalpha()):   # isalpha() 方法检测字符串是否只由字母组成
            #print("1")
            continue
        if len(set(pep).difference(set('ARNDCQEGHILKMFPSTWYV'))) != 0:  #存在异常氨基酸
            #print("2")
            continue
        if len(set(hla_seq).difference(set('ARNDCQEGHILKMFPSTWYV'))) != 0:
            #print("3")
            continue
        if len(hla_seq) > 34:
            #print("8")
            continue

        length = len(pep)
        print("pep的长度：",length)
        if length <= 15:
            print("123")
            print("cut_peptide:",cut_peptide)
            print("cut_peptide类型:",type(cut_peptide))
            if cut_peptide:   #进行切割
                print("1234")
                if length > cut_length:
                    print("4")
                    cut_peptides = [pep] + [pep[i : i + cut_length] for i in range(length - cut_length + 1)]
                    peptides.extend(cut_peptides)
                    HLA_sequences.extend([hla_seq] * len(cut_peptides))  #添加多个HLA序列，否则pep和HLA的数量不一样会报错
                    HLA_names.extend([hla_name] * len(cut_peptides))
                else:
                    print("5")
                    peptides.append(pep)
                    HLA_sequences.append(hla_seq)
                    HLA_names.append(hla_name)
            else:
                print("6")
                peptides.append(pep)
                HLA_sequences.append(hla_seq)
                HLA_names.append(hla_name)
                
        else:   #长度过长，超出我们的最大预测长度15
            print("7")
            cut_peptides = [pep[i : i + cut_length] for i in range(length - cut_length + 1)]
            peptides.extend(cut_peptides)
            HLA_sequences.extend([hla_seq] * len(cut_peptides))
            HLA_names.extend([hla_name] * len(cut_peptides))
    
    print("length of HLA_names:", len(HLA_names)) 
    print("length of HLA_sequences:", len(HLA_sequences))
    print("length of peptides:", len(peptides))

    predict_data = pd.DataFrame([HLA_names, HLA_sequences, peptides], index = ['HLA', 'HLA_sequence', 'peptide']).T
    #print(predict_data.shape)
    if predict_data.shape[0] == 0:   #输入数据有问题
        print('No suitable data could be predicted. Please check your input data.')
        return "" # 返回空不退出程序

    batch_size = 1024  
    predict_data, predict_pep_inputs, predict_hla_inputs, predict_loader = models.read_predict_data(predict_data, batch_size)

    # # 预测

    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    model_file = './model/transbindpmhc/model_layer5_multihead1_fold4.pkl'

    model_eval = models.Transformer(n_layers=5,n_heads=1,n_fold=4).to(device)
    model_eval.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')), strict = True)  #修改这里使其用cpu加载

    model_eval.eval()
    y_pred, y_prob, attns = models.eval_step(model_eval, predict_loader, threshold, use_cuda)

    predict_data['y_pred'], predict_data['y_prob'] = y_pred, y_prob
    predict_data = predict_data.round({'y_prob': 4})


    return predict_data

if __name__ == '__main__':
    #peptide_input = "./examples/peptides_2.fasta"
    #peptide_input = "AEAFIQPI"
    #HLA_input = "./examples/hlas_2.fasta"
    #HLA_input = "HLA-A*11:01"
    
    #gene = "APC"
    # peptide_input = f"./examples/peptides_share_{gene}_neoantigen.fasta"
    # HLA_input = f"./examples/hlas_share_{gene}_neoantigen.fasta"
    peptide_input = f"./examples/peptides_2024-05-28.fasta"
    HLA_input = f"./examples/hlas_all.fasta"
    predict_data = predict(peptide_input, HLA_input, threshold=0.5, cut_peptide=False, cut_length=9)
    predict_data.to_csv(f"./examples/results_2024.05.27.csv", index=None)



