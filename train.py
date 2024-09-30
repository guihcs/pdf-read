import math
import json

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import base64
import io
from PIL import Image
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import cv2
from tqdm.auto import tqdm
import torch.utils.checkpoint as checkpoint
from model import VitXl, PretrainVit, PDFReader

torch.backends.cuda.enable_flash_sdp(True)
# set white background in matplotlib
plt.rcParams['figure.facecolor'] = 'white'

# data_path = '/projets/melodi/gsantoss/pages.json'
data_path = '/projets/melodi/gsantoss/pdf/dataset.json'

with open(data_path) as f:
    data = json.load(f)


def slice_image(img, size=16, stride=16):
    p_imag = img.unfold(0, size, stride).unfold(1, size, stride)
    return torch.flatten(p_imag, 0, end_dim=1)


tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

imgs = []
texts = []

for doc_image, doc_text in tqdm(data):
    # doc_image, doc_text = d['image'], d['text']
    try:
        img = base64.b64decode(doc_image[22:])
        img = torch.from_numpy(np.array(Image.open(io.BytesIO(img))))

        # convert image to grayscale using cv2
        img = cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2GRAY)
        img = torch.from_numpy(img) / 255.0

        # resize image to 517x400
        img = nn.functional.interpolate(img.unsqueeze(0).unsqueeze(0), size=(517, 400), mode='bilinear',
                                        align_corners=False).squeeze(0).squeeze(0)
        imgs.append(torch.flatten(slice_image(img, size=16, stride=16), start_dim=1).unsqueeze(0))

        texts.append('<s>' + doc_text)
    except:
        break
        pass

tokens = tokenizer(texts, return_tensors='pt', padding=True)
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']
print(input_ids.shape, attention_mask.shape)
dataset = TensorDataset(torch.cat(imgs, dim=0).float(), input_ids, attention_mask)


def model_eval(tokenizer, model, data, batch_size=32, ml=512):
    pred = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, ym, ya in DataLoader(data, batch_size=batch_size):
            res = model.read(tokenizer, x.cuda(0), ml=ml).cpu()
            comp = res == ym[:, :ml]
            pred += torch.sum(comp.float() * (ym[:, :ml] != 0).float())
            total += torch.sum((ym[:, :ml] != 0).float())

    model.train()
    return pred / total



vit = VitXl(tokenizer.vocab_size, 256, 512, 4, 1024, 2, max_len=800, seq_len=512, mem_len=64)
vit.load_state_dict(torch.load('/projets/melodi/gsantoss/models/pdfre.pt'))

model = PDFReader(vit, tokenizer.vocab_size, d_model=512, dim_feedforward=1024)

model = nn.DataParallel(model)
model.cuda(0)

crit = nn.NLLLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.00003)
batch_size = 32
epochs = 300
lh = []
th = []
test = TensorDataset(*dataset[:100])
train = TensorDataset(*dataset[200:])
val = TensorDataset(*dataset[100:200])
progress = tqdm(total=epochs * math.ceil(len(train) / batch_size))

compiled = model

for e in range(epochs):
    el = []
    model.train()
    for x, y, ya in DataLoader(train, batch_size=batch_size, shuffle=True):
        optimizer.zero_grad()
        res = compiled(x.cuda(0), y.long().cuda(0))
        rf = torch.flatten(res[:, :-1, :], start_dim=0, end_dim=1)
        yf = torch.flatten(y[:, 1:], start_dim=0, end_dim=1).cuda(0)
        loss = crit(rf, yf) * (yf != 0).float()
        loss = loss.sum() / (yf != 0).float().sum()
        loss.backward()
        el.append(loss.item())
        optimizer.step()
        progress.update(1)

    lh.append(sum(el) / len(el))
    if e % 100 == 0:
        th.append(model_eval(tokenizer, compiled, val, ml=512))

progress.close()
fig, ax = plt.subplots(1, 2)
fig.tight_layout()
ax[0].plot(lh)
ax[1].plot(th, c='g')
plt.show()

torch.save(model.module.state_dict(), '/projets/melodi/gsantoss/models/pdfr.pt' )
print(f'loss: {lh[-1]:0.2f}, acc: {th[-1]:0.2f}')