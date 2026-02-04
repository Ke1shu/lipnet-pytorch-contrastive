import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ouludataset import *
from model import LipNet


opt = __import__('options')
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu



def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
    return DataLoader(dataset,
        batch_size = opt.batch_size, 
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False)




model = LipNet()
model = model.cuda()
net = nn.DataParallel(model).cuda()



# 2. 学習済みの重みを読み込む
model.load_state_dict(torch.load('./weights/LipNet_unseen_loss_2.7482646081758584_wer_1.0707435904272438_cer_2.3712408762952655.pt'))
model.eval()  # 評価モードに設定



inputs = []

for i in range(10):
      
      dataset = PhraseSort(i,opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train')
      
      loader = dataset2dataloader(dataset)
      iter_loader = iter(loader)
      input = next(iter_loader)
      
      inputs.append(input)


input = inputs[1]

      
vids = input.get('vids')
vid1 , vid2 = vids #vid1 横顔、vid2 正面
vid1 = vid1.cuda()
vid2 = vid2.cuda()
txt = input.get('txt').cuda()
#vid_len = input.get('vid_len').cuda()
#txt_len = input.get('txt_len').cuda()
                
y1 = net(vid1) #[vid_len, batch , gru 512]
y2 = net(vid2)


# UMAPに入力するために、時間ステップTのすべてのバッチの特徴量を抽出
# 例えば、時間ステップt=10の特徴量を抽出
t = 10
features_at_t = y1[:, t, :].detach().cpu().numpy()  # shape: (B, 28)

# UMAPによる次元削減
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(features_at_t)  # shape: (B, 2)


# プロット
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], s=50, cmap='Spectral')
plt.title(f'UMAP projection of features at time step {t}')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.show()