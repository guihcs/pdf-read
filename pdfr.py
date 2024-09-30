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
from model import VitXl, PretrainVit

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

vit = VitXl(tokenizer.vocab_size, 256, 512, 4, 1024, 2, max_len=800, seq_len=512, mem_len=64)
pretrain = PretrainVit(vit, tokenizer.vocab_size, v_dim=256, n_dim=512)

# checkpoint = torch.load('/projets/melodi/gsantoss/models/checkpoint_pdfre.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
pretrain = nn.DataParallel(pretrain)
pretrain.cuda(0)
pretrain.train()
crit1 = nn.BCELoss()
crit2 = nn.NLLLoss(reduction='none')

optimizer = optim.Adam(pretrain.parameters(), lr=0.00003)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
batch_size = 16
epochs = 150
progress = tqdm(total=epochs * math.ceil(len(dataset) / batch_size))
lowest_loss = float('inf')
lh = []
epoch = 0

compiled = pretrain

# lh = checkpoint['loss']
# epoch = checkpoint['epoch']

progress.update(epoch * math.ceil(len(dataset) / batch_size))

for epoch in range(epoch, epochs):
    el = []
    for x, y, ya in DataLoader(dataset, batch_size=batch_size, shuffle=True):
        optimizer.zero_grad()

        with torch.no_grad():
            vn = torch.randn_like(x).cuda(0)
            tn = torch.randint_like(y, 0, tokenizer.vocab_size).cuda(0)

            vnm = torch.rand(x.shape[0], x.shape[1]) < 0.15
            tnm = torch.rand(y.shape[0], y.shape[1]) < 0.15

            vn *= vnm.unsqueeze(-1).float().cuda(0)
            tn *= tnm.long().cuda(0)

            x = torch.clamp(x.cuda(0) + vn, 0, 1)
            y = torch.clamp(y.cuda(0) + tn, 0, tokenizer.vocab_size - 1)

            vn = torch.rand(x.shape[0], x.shape[1]) > 0.85

            x *= vn.unsqueeze(-1).float().cuda(0)

        with torch.cuda.amp.autocast():
            r1, r2 = compiled(x.cuda(0), y.long().cuda(0))

        l1 = crit1(r1.float().cpu(), x.cpu())

        rf = torch.flatten(r2[:, :-1, :].float().cpu(), start_dim=0, end_dim=1)
        yf = torch.flatten(y[:, 1:], start_dim=0, end_dim=1).cpu()
        l2 = crit2(rf, yf) * (yf != 0).float()
        l2 = l2.sum() / (yf != 0).float().sum()

        loss = l1 + l2

        loss.backward()
        el.append(loss.item())
        optimizer.step()
        progress.update(1)

    lh.append(sum(el) / len(el))
    if lh[-1] < lowest_loss:
        lowest_loss = lh[-1]
        torch.save({
            'epoch': epoch,
            'model_state_dict': pretrain.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': lh,
        }, '/projets/melodi/gsantoss/models/checkpoint_pdfre.pt')
        torch.save(pretrain.module.vit.state_dict(), '/projets/melodi/gsantoss/models/pdfreb.pt')

progress.close()
fig, ax = plt.subplots(1, 2)
fig.tight_layout()
ax[0].plot(lh)
plt.show()

if lh[-1] < lowest_loss:
    torch.save({
        'epoch': epoch,
        'model_state_dict': pretrain.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': lh,
    }, '/projets/melodi/gsantoss/models/checkpoint_pdfre.pt')

    torch.save(pretrain.module.vit.state_dict(), '/projets/melodi/gsantoss/models/pdfreb.pt')
print(f'loss: {lh[-1]:0.2f}')