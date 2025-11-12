### Comandos para executar o treinamento do modelo de análise se sentimentos:
# pip install matplotlib numpy pandas plotly torch scikit-learn transformers tqdm
# python treinamento_analise_sentimentos.py
# O dataset utilizado para o treinamento pode ser encontrado no seguinte link: https://www.kaggle.com/code/viniciuscleves/an-lise-de-sentimento-com-bert/input

# Importação das bibliotecas necessárias
import math
import os
import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

# Carregamento do conjunto de dados
data = pd.read_csv('imdb-reviews-pt-br.csv', index_col=0)
data.head(1)

# Informações sobre o conjunto de dados
data.info()

# Visualização da distribuição das classes no conjunto de dados
data['sentiment'].value_counts().plot.pie(autopct='%.2f', explode=[0.01, 0])

# Divisão do conjunto de dados em treino, validação e teste
test_dev_size = int(0.05 * data.shape[0])
train_dev, test = train_test_split(data, test_size=test_dev_size, random_state=42, stratify=data['sentiment'])
train, dev = train_test_split(train_dev, test_size=test_dev_size, random_state=42, stratify=train_dev['sentiment'])
print('Training samples:', train.shape[0])
print('Dev samples:     ', dev.shape[0])
print('Test samples:    ', test.shape[0])

# Definição da classe do conjunto de dados customizado
class ImdbPt(Dataset):
    ''' Loads IMDB-pt dataset.

    It will tokenize our inputs and cut-off sentences that exceed 512 tokens (the pretrained BERT limit)
    '''
    def __init__(self, tokenizer, X, y):
        X = list(X)
        y = list(y)
        tokenized_data = tokenizer(X, truncation=True, max_length=512)
        samples = [
            {
                **{key: tokenized_data[key][i] for key in tokenized_data},
                'labels': y[i]
            }

            for i in range(len(X))
        ]
        self.samples = samples

    def __getitem__(self, i):
        return self.samples[i]

    def __len__(self):
        return len(self.samples)

# Função para enviar inputs para o dispositivo (CPU/GPU)
def send_inputs_to_device(inputs, device):
    return {key: tensor.to(device) for key, tensor in inputs.items()}

# Inicialização do tokenizador BERT em português
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

# Criação dos conjuntos de dados de treino, validação e teste
train_dataset = ImdbPt(tokenizer, train['text_pt'], (train['sentiment'] == 'pos').astype(int))
dev_dataset = ImdbPt(tokenizer, dev['text_pt'], (dev['sentiment'] == 'pos').astype(int))
test_dataset = ImdbPt(tokenizer, test['text_pt'], (test['sentiment'] == 'pos').astype(int))

# Criação dos DataLoader para treino, validação e teste
train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=DataCollatorWithPadding(tokenizer))
dev_loader = DataLoader(dev_dataset, batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer))
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer))

# Inicialização do modelo BERT para classificação de sequências
model = BertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased')
model.train()

# Configuração do dispositivo de treino (GPU se disponível, caso contrário, CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Configuração do otimizador e agendador de taxa de aprendizado
optimizer = AdamW(model.parameters(), lr=5e-6)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9997)

# Congelamento dos parâmetros do modelo base (BERT)
for param in model.base_model.parameters():
    param.requires_grad = False

# Função para avaliação do modelo na fase de validação
def evaluate(model, dev_loader, device):
    with torch.no_grad():
        model.eval()
        dev_losses = []
        tp, tn, fp, fn = [], [], [], []
        for inputs in dev_loader:
            inputs = send_inputs_to_device(inputs, device)
            loss, scores = model(**inputs)[:2]
            dev_losses.append(loss.cpu().item())

            _, classification = torch.max(scores, 1)
            labels = inputs['labels']
            tp.append(((classification == 1) & (labels == 1)).sum().cpu().item())
            tn.append(((classification == 0) & (labels == 0)).sum().cpu().item())
            fp.append(((classification == 1) & (labels == 0)).sum().cpu().item())
            fn.append(((classification == 0) & (labels == 1)).sum().cpu().item())

        tp_s, tn_s, fp_s, fn_s = sum(tp), sum(tn), sum(fp), sum(fn)
        print('Dev loss: {:.2f}; Acc: {:.2f}; tp: {}; tn: {}; fp: {}; fn: {}'.format(
            np.mean(dev_losses), (tp_s + tn_s) / (tp_s + tn_s + fp_s + fn_s), tp_s, tn_s, fp_s, fn_s))

        model.train()

# Treinamento do modelo
epoch_bar = tqdm_notebook(range(1))
loss_acc = 0
alpha = 0.95
for epoch in epoch_bar:
    batch_bar = tqdm_notebook(enumerate(train_loader), desc=f'Epoch {epoch}', total=len(train_loader))
    for idx, inputs in batch_bar:
        if (epoch * len(train_loader) + idx) == 800:
            for param in model.base_model.parameters():
                param.requires_grad = True

        inputs = send_inputs_to_device(inputs, device)
        optimizer.zero_grad()
        loss, logits = model(**inputs)[:2]

        loss.backward()
        optimizer.step()
        if epoch == 0 and idx == 0:
            loss_acc = loss.cpu().item()
        else:
            loss_acc = loss_acc * alpha + (1 - alpha) * loss.cpu().item()
        batch_bar.set_postfix(loss=loss_acc)
        if idx % 200 == 0:
            del inputs
            del loss
            evaluate(model, dev_loader, device)

        scheduler.step()
    os.makedirs('/working/checkpoints/epoch' + str(epoch))
    model.save_pretrained('/working/checkpoints/epoch' + str(epoch))

# Avaliação do modelo na fase de teste
with torch.no_grad():
    model.eval()
    pred = []
    labels = []
    for inputs in tqdm_notebook(test_loader):
        inputs = send_inputs_to_device(inputs, device)
        _, scores = model(**inputs)[:2]
        pred.append(F.softmax(scores, dim=1)[:, 1].cpu())
        labels.append(inputs['labels'].cpu())
pred = torch.cat(pred).numpy()
labels = torch.cat(labels).numpy()

# Cálculo da curva ROC e apresentação visual
fpr, tpr, thresholds = metrics.roc_curve(labels, pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

fig = px.scatter(
    x=fpr, y=tpr, color=thresholds,
    labels={'x': 'False positive rate', 'y': 'True positive rate'},
    title='Curva ROC')
fig.show()

# Avaliação da acurácia em diferentes thresholds
acc = []
for th in thresholds:
    acc.append(metrics.accuracy_score(pred > th, labels))

fig2 = px.scatter(
    x=thresholds, y=acc, labels={'x': 'threshold', 'y': 'acurácia'},
    title='Acurácia em diferentes thresholds')
fig2.show()

# Avaliação final na fase de teste
with torch.no_grad():
    model.eval()
    pred = []
    labels = []
    for inputs in tqdm_notebook(test_loader):
        inputs = send_inputs_to_device(inputs, device)
        _, scores = model(**inputs)[:2]
        pred.append(F.softmax(scores, dim=1)[:, 1].cpu())
        labels.append(inputs['labels'].cpu())
pred = torch.cat(pred).numpy()
labels = torch.cat(labels).numpy()

# Avaliação da acurácia com um threshold específico
print('Acc:', metrics.accuracy_score(pred > 0.67, labels))

# Salvando o modelo treinado
model_save_path = 'Modelo'
os.makedirs(model_save_path, exist_ok=True)
model.save_pretrained(model_save_path)

# Importação do modelo treinado para realizar a classificação de sentimentos em frases
from transformers import BertForSequenceClassification, BertTokenizer

# Carregamento do modelo e tokenizador treinados
model_path = 'Modelo'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

# Exemplo de classificação de sentimento em uma frase personalizada
texto_personalizado = "Este é um ótimo exemplo de como usar o modelo de linguagem."
inputs = tokenizer(texto_personalizado, return_tensors="pt")

# Inferência do modelo na frase personalizada
with torch.no_grad():
    model.eval()
    scores = model(**inputs)[0]

# Cálculo da probabilidade da classe positiva
prob_pos = F.softmax(scores, dim=1)[:, 1]

# Definição de um limiar de probabilidade para a classificação
limiar_probabilidade = 0.5

# Classificação final com base no limiar
if prob_pos > limiar_probabilidade:
    classe_resultante = "Positivo"
else:
    classe_resultante = "Negativo"

# Impressão do resultado da classificação
print(f"Texto classificado como: {classe_resultante}")
print(f"Probabilidade da classe positiva: {prob_pos.item()}")