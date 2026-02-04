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

# 学習済みの重みを読み込む
model.load_state_dict(torch.load('./weights/LipNet_unseen_loss_2.7482646081758584_wer_1.0707435904272438_cer_2.3712408762952655.pt'))
model.eval()  # 評価モードに設定

all_features = []
all_labels = []

# ラベルの定義
label_texts = [
    "Excuse me", "Goodbye", "Hello", "How are you", "Nice to meet you",
    "See you", "I am sorry", "Thank you", "Have a good time", "You are welcome"
]

for i in range(10):
    dataset = PhraseSort(i, opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train')
    
    loader = dataset2dataloader(dataset)
    iter_loader = iter(loader)
    input = next(iter_loader)
    
    vids = input.get('vids')
    vid1, vid2 = vids  # vid1 横顔、vid2 正面
    vid1 = vid1.cuda()
    vid2 = vid2.cuda()
    
    with torch.no_grad():
        y1 = net(vid1)  # [vid_len, batch, gru 512]
        y2 = net(vid2)
    
    # 時間ステップt=10の特徴量を抽出
    t = 2
    features_at_t = y1[:, t, :].detach().cpu().numpy()  # shape: (B, 512)
    
    all_features.append(features_at_t)
    all_labels.extend([label_texts[i]] * features_at_t.shape[0])

# 全ての特徴量を結合
all_features = np.vstack(all_features)
all_labels = np.array(all_labels)

# UMAPによる次元削減
#reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
reducer = umap.UMAP(n_neighbors=100,min_dist=0.9,metric='cosine', n_components=2, random_state=42)
embedding = reducer.fit_transform(all_features)

# プロット
plt.figure(figsize=(16, 12))
unique_labels = np.unique(all_labels)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
for label, color in zip(unique_labels, colors):
    mask = all_labels == label
    plt.scatter(embedding[mask, 0], embedding[mask, 1], c=[color], label=label, s=50)

plt.title(f'UMAP projection of features at time step {t} for 10 phrases')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()