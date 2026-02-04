import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
from ouludataset import *
import numpy as np
import time
from model import LipNet
import torch.optim as optim
import re
import json
from tensorboardX import SummaryWriter

import csv
import datetime
import wandb

#textgrid
import textgrid
from collections import defaultdict



if(__name__ == '__main__'):
    opt = __import__('options')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu    
    writer = SummaryWriter()

    train_logs = {'loss':[],'wer':[],'cer':[]}  # lossを保存するリスト
    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="lipnet-pytorch",

        # track hyperparameters and run metadata
        config={
            "hoge" : 0
        }
    )

def custom_collate_fn(batch):
    # TextGridオブジェクトをそのままリストに格納して返す

    # vids の各ビデオを分けてスタックする
    vid1_batch = torch.stack([item['vids'][0] for item in batch])
    vid2_batch = torch.stack([item['vids'][1] for item in batch])

    collated_batch = {
        'vids': (vid1_batch, vid2_batch),  # (vid1, vid2) の形にする
        'txt': torch.nn.utils.rnn.pad_sequence([item['txt'] for item in batch], batch_first=True),
        'txt_len': torch.tensor([item['txt_len'] for item in batch]),
        'vid_len': torch.tensor([item['vid_len'] for item in batch]),
        'phonem': [item['phonem'] for item in batch]  # TextGrid オブジェクトをリストのまま保持
    }
    return collated_batch



def textgrid2dic(textgrid):

    #fps 動画のフレームレート
    fps=29.97


    phoneme_intervals = defaultdict(lambda: [[] for _ in range(len(textgrid))])

    # 各バッチの音素を確認して辞書に区間を格納
    for j, tg in enumerate(textgrid):
        for interval in tg[1].intervals:
            phoneme_label = interval.mark  # 音素ラベル
            # バッチの該当位置に minTime と maxTime をタプルとしてリストに追加

            start = int(interval.minTime * fps)
            end = int(interval.maxTime * fps)
            phoneme_intervals[phoneme_label][j].append((start, end))
    
    return phoneme_intervals

        



def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
    #textgridオブジェクトをそのまま返すようにcustom_collate_fn関数を指定
    return DataLoader(dataset,
        batch_size = opt.batch_size, 
        collate_fn=custom_collate_fn,
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False)

def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()  

def ctc_decode(y):
    result = []
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]


def contrastive_loss(z1, z2, temperature=0.1):
    z1 = z1.permute(1, 0, 2).contiguous()
    z2 = z2.permute(1, 0, 2).contiguous()
    
    # zi, zj: [T, B, D] where T is time steps, B is batch size, D is feature dimension
    T, B, D = z1.shape
    
    # 特徴量の正規化
    z1_norm = F.normalize(z1, dim=2)
    z2_norm = F.normalize(z2, dim=2)

    # Reshape to [T*B, D]
    z1_flat = z1_norm.reshape(-1, D)
    z2_flat = z2_norm.reshape(-1, D)

    # Concatenate along the batch dimension
    representation = torch.cat([z1_flat, z2_flat], dim=0)  # [2*T*B, D]

    #representation = torch.cat([zi_norm, zj_norm], dim=0)  # [2*batch_size, h_dim]
    sim = torch.matmul(representation, torch.t(representation)) #積
    sim_ij = torch.diag(sim, T*B) #対角成分を取り出す
    sim_ji = torch.diag(sim, -T*B) #対角成分を取り出す
    top = torch.exp(torch.cat([sim_ij, sim_ji], dim=0)/temperature) #positive_pairの積の対角成分をtauで割ってexp
    mask = torch.ones((2*T*B, 2*T*B), dtype=bool) #マスクの作成、対角成分だけfalse
    for i in range(T*B):
        mask[i, i] = False
    mask = mask.cuda()
    bot = mask * torch.exp(sim / temperature)  # 要素積, positive negativeの積
    all_losses = - torch.log(top/torch.sum(bot, dim=1))
    loss = torch.sum(all_losses) / (2*T*B)
    return loss



def contrastive_phonem_loss(z1, z2, phoneme_dict, temperature=0.1):
    z1 = z1.permute(1, 0, 2).contiguous()
    z2 = z2.permute(1, 0, 2).contiguous()
    
    T, B, D = z1.shape

    # 正規化
    z1_norm = F.normalize(z1, dim=2)
    z2_norm = F.normalize(z2, dim=2)

    # 2つの視点の特徴量をフラット化
    z1_flat = z1_norm.reshape(-1, D)
    z2_flat = z2_norm.reshape(-1, D)

    # 全ての表現をまとめる
    representation = torch.cat([z1_flat, z2_flat], dim=0)

    # 類似度行列の作成
    sim = torch.matmul(representation, representation.t())

    # マスクの作成
    mask = torch.zeros_like(sim, dtype=bool)

    for phoneme, frame_lists in phoneme_dict.items():
        for batch_index, frame_ranges in enumerate(frame_lists):
            # すべてのフレーム区間間の組み合わせを正例ペアとして設定
            for (start1, end1) in frame_ranges:
                for (start2, end2) in frame_ranges:
                    # 各フレーム範囲内での全組み合わせをマスクに設定
                    for i in range(start1, end1 + 1):
                        for j in range(start2, end2 + 1):
                            # 正例ペアを設定
                            mask[i + batch_index * T, j + batch_index * T] = True
                            mask[i + batch_index * T, j + batch_index * T + T * B] = True
                            mask[i + batch_index * T + T * B, j + batch_index * T] = True
                            mask[i + batch_index * T + T * B, j + batch_index * T + T * B] = True

    # 正例ペアの類似度と全体の類似度を取得
    positive_exp = torch.exp(sim[mask] / temperature)
    denominator = torch.exp(sim / temperature).sum(dim=1, keepdim=True)

    # maskを用いて正例ペアの指数を全体の分母に合わせる
    mask_sum = mask.sum(dim=1, keepdim=True)
    all_losses = -torch.log(positive_exp / denominator[mask_sum])

    # 損失値の平均を計算
    loss = all_losses.mean()
    
    return loss

def test(model, net, lossfunc):

    with torch.no_grad():
        dataset = FrontProfile(opt.video_path,
                               opt.anno_path,
                               opt.train_list,
                               opt.vid_padding,
                               opt.txt_padding,
                               'test')

        print('num_test_data:{}'.format(len(dataset.data)))
        model.eval()
        loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        wer = []
        cer = []
        crit = nn.CTCLoss()
        tic = time.time()
        for (i_iter, input) in enumerate(loader):            
            vids = input.get('vids')
            vid1, vid2 = vids
            vid1 = vid1.cuda()
            if lossfunc == contrastive_phonem_loss:
                # textgrid
                tg = input.get('phonem')
                tg = textgrid2dic(tg)
                vid2 = vid2.cuda()
            
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()

            y = net(vid1)
            if lossfunc == contrastive_phonem_loss:
                y2 = net(vid2)
                loss = contrastive_phonem_loss(y, y2, tg)
            else:
                loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))

            # detach and move loss to CPU before converting to numpy
            loss_list.append(loss.detach().cpu().numpy())

            pred_txt = ctc_decode(y)
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]

            wer.extend(MyDataset.wer(pred_txt, truth_txt)) 
            cer.extend(MyDataset.cer(pred_txt, truth_txt))

            if i_iter % opt.display == 0:
                v = 1.0 * (time.time() - tic) / (i_iter + 1)
                eta = v * (len(loader) - i_iter) / 3600.0

                print(''.join(101 * '-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101 * '-'))
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101 * '-'))
                print('test_iter={}, eta={}, wer={}, cer={}'.format(
                    i_iter, eta,
                    np.array(wer).mean(),
                    np.array(cer).mean()
                ))
                print(''.join(101 * '-'))

        # Ensure wer and cer are converted to numpy after moving to CPU
        wer = np.array(wer)
        cer = np.array(cer)
        return np.array(loss_list).mean(), wer.mean(), cer.mean()



'''
def test(model, net,testloss):

    if testloss=="contrastive":
        contrastiveBool = True
    else:
        contrastiveBool = False

    with torch.no_grad():
        
        dataset = FrontProfile(opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'test')
            
        print('num_test_data:{}'.format(len(dataset.data)))  
        model.eval()
        loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        wer = []
        cer = []
        crit = nn.CTCLoss()
        tic = time.time()
        for (i_iter, input) in enumerate(loader):            
            vids = input.get('vids')
            vid1 , vid2 = vids
            vid1 = vid1.cuda()
            if contrastiveBool:
                #textgrid
                tg = input.get('phonem')
                tg = textgrid2dic(tg)
                vid2 = vid2.cuda()
            
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()

            
            y = net(vid1)
            if contrastiveBool:
                y2 = net(vid2)
                loss = contrastive_phonem_loss(y,y2,tg)
            else:
                loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
            loss_list.append(loss)
            pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]

            wer.extend(MyDataset.wer(pred_txt, truth_txt)) 
            cer.extend(MyDataset.cer(pred_txt, truth_txt))              
            if(i_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(loader)-i_iter) / 3600.0
                
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    print('{:<50}|{:>50}'.format(predict, truth))                
                print(''.join(101 *'-'))
                print('test_iter={},eta={},wer={},cer={}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean()))                
                print(''.join(101 *'-'))
                
        return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())
    '''
    
def pre_train(model, net):

    now = datetime.datetime.now()
    # ファイル名に日付と時間を組み込む
    nowstr = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    dataset = FrontProfile(opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train')
        
    loader = dataset2dataloader(dataset) 
    optimizer = optim.Adam(model.parameters(),
                lr = opt.base_lr,
                weight_decay = 0.,
                amsgrad = True)
                
    print('num_train_data:{}'.format(len(dataset.data)))    
    crit = nn.CTCLoss()
    tic = time.time()
    
    train_wer = []
    train_cer = []
    
    for epoch in range(opt.max_epoch):
        loss_sum = 0
        cer_sum = 0
        wer_sum = 0
        for (i_iter, input) in enumerate(loader):
            model.train()
            
            vids = input.get('vids')
            vid1 , vid2 = vids
            vid1 = vid1.cuda()
            vid2 = vid2.cuda()
            txt = input.get('txt').cuda()
            #vid_len = input.get('vid_len').cuda()
            #txt_len = input.get('txt_len').cuda()

            #textgrid
            tg = input.get('phonem')
            tg = textgrid2dic(tg)

            
            optimizer.zero_grad()
            
            y1 = net(vid1) #[vid_len, batch , gru 512]
            y2 = net(vid2)

            y = net(vid1)



            
            #loss = contrastive_loss(y1,y2)
            loss = contrastive_phonem_loss(y1,y2,tg)
            loss_sum += loss.item()
            loss.backward()
            if(opt.is_optimize):
                optimizer.step()

            
            
            tot_iter = i_iter + epoch*len(loader)
            
            pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))

            cer_sum += np.array(train_cer).mean()
            wer_sum += np.array(train_wer).mean()
            
            
            

            
            
            if(tot_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(loader)-i_iter)*v/3600.0
                
                writer.add_scalar('train loss', loss, tot_iter)
                writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)              
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))                
                print(''.join(101*'-'))
                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101*'-'))                
                print('epoch={},tot_iter={},eta={},loss={},train_wer={}'.format(epoch, tot_iter, eta, loss, np.array(train_wer).mean()))
                print(''.join(101*'-'))
                
            if(tot_iter % opt.test_step == 0):                
                (loss, wer, cer) = test(model, net,contrastive_phonem_loss)
                wandb.log({
                    "val_loss" : loss,
                    "val_cer" : cer,
                    "val_wer" : wer
                })
                print('i_iter={},lr={},loss={},wer={},cer={}'
                    .format(tot_iter,show_lr(optimizer),loss,wer,cer))
                writer.add_scalar('val loss', loss, tot_iter)                    
                writer.add_scalar('wer', wer, tot_iter)
                writer.add_scalar('cer', cer, tot_iter)
                savename = '{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.save_prefix, loss, wer, cer)
                (path, name) = os.path.split(savename)
                if(not os.path.exists(path)): os.makedirs(path)
                #torch.save(model.state_dict(), savename)
                if(not opt.is_optimize):
                    exit()
        epoch_loss = loss_sum / len(loader)  # 1epochのlossを計算
        epoch_cer = cer_sum / len(loader)
        epoch_wer = wer_sum / len(loader)

        wandb.log({
            "epoch" : epoch,
            "train_loss" : epoch_loss,
            "train_cer" : epoch_cer,
            "train_wer" : epoch_wer
        })
        
        
        
        savename = '{}_{}_loss_{}_wer_{}_cer_{}.pt'.format(nowstr,opt.save_prefix, epoch_loss, epoch_wer, epoch_cer)
        (path, name) = os.path.split(savename)
        if(not os.path.exists(path)): os.makedirs(path)
        #動作テスト中は消しておきます
        torch.save(model.state_dict(), savename)

        print('loss : {} cer: {}'.format(epoch_loss,epoch_cer))
        train_logs['loss'].append(epoch_loss)
        train_logs['cer'].append(epoch_cer)
        train_logs['wer'].append(epoch_wer)
    
    
    csvname = f'logs/LipNet_{nowstr}.csv'
    with open(csvname, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # ヘッダーを書き込む
        csvwriter.writerow(['epoch', 'loss', 'wer', 'cer'])
        # 各エポックのデータを書き込む
        for epoch in range(len(train_logs['loss'])):
            loss = train_logs['loss'][epoch]
            wer = train_logs['wer'][epoch]
            cer = train_logs['cer'][epoch]
            csvwriter.writerow([epoch, loss, wer, cer])

#valのlossを記録できるように変更
def pre_train2(model, net):
    now = datetime.datetime.now()
    nowstr = now.strftime("%Y-%m-%d_%H-%M-%S")  # ファイル名に日付と時間を組み込む
    
    dataset = FrontProfile(
        opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train'
    )
    
    loader = dataset2dataloader(dataset)
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.base_lr,
        weight_decay=0.0,
        amsgrad=True
    )
    
    print(f'num_train_data: {len(dataset.data)}')
    crit = nn.CTCLoss()
    tic = time.time()

    train_logs = {'loss': [], 'cer': [], 'wer': []}
    csvname = f'logs/LipNet_{nowstr}.csv'

    # CSVヘッダーの書き込み
    with open(csvname, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['epoch', 'loss', 'wer', 'cer'])

    for epoch in range(opt.max_epoch):
        loss_sum, cer_sum, wer_sum = 0, 0, 0
        train_wer, train_cer = [], []

        val_epoch = 0

        for i_iter, input in enumerate(loader):
            model.train()

            vids = input.get('vids')
            vid1, vid2 = vids[0].cuda(), vids[1].cuda()
            txt = input.get('txt').cuda()
            tg = textgrid2dic(input.get('phonem'))

            optimizer.zero_grad()
            y1 = net(vid1)
            y2 = net(vid2)
            y = net(vid1)

            # 計算とバックプロパゲーション
            loss = contrastive_phonem_loss(y1, y2, tg)
            loss_sum += loss.item()
            loss.backward()
            if opt.is_optimize:
                optimizer.step()

            # デコードと評価
            pred_txt = ctc_decode(y)
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))

            # ログ
            tot_iter = i_iter + epoch * len(loader)
            if tot_iter % opt.display == 0:
                avg_wer = np.mean(train_wer)
                avg_cer = np.mean(train_cer)
                writer.add_scalar('train loss', loss.item(), tot_iter)
                writer.add_scalar('train wer', avg_wer, tot_iter)
                writer.add_scalar('train cer', avg_cer, tot_iter)
                print('-' * 101)
                print(f'epoch={epoch}, tot_iter={tot_iter}, loss={loss.item()}, train_wer={avg_wer}')
                print('-' * 101)
            
            if tot_iter % opt.test_step == 0:
                val_loss, val_wer, val_cer = test(model, net, contrastive_phonem_loss)
                
                wandb.log({
                    "val_loss": val_loss,
                    "val_cer": val_cer,
                    "val_wer": val_wer
                }, step=val_epoch)
                val_epoch += 1
                writer.add_scalar('val loss', val_loss, tot_iter)
                writer.add_scalar('val wer', val_wer, tot_iter)
                writer.add_scalar('val cer', val_cer, tot_iter)

        # エポック終了後の統計
        epoch_loss = loss_sum / len(loader)
        epoch_cer = np.mean(train_cer)
        epoch_wer = np.mean(train_wer)
        train_logs['loss'].append(epoch_loss)
        train_logs['cer'].append(epoch_cer)
        train_logs['wer'].append(epoch_wer)


        # エポック単位のログを記録
        wandb.log({
            "train_loss": epoch_loss,
            "train_cer": epoch_cer,
            "train_wer": epoch_wer
        }, step=epoch)

        with open(csvname, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([epoch, epoch_loss, epoch_wer, epoch_cer])

        # モデル保存
        savename = f'{nowstr}_{opt.save_prefix}_loss_{epoch_loss:.4f}_wer_{epoch_wer:.4f}_cer_{epoch_cer:.4f}.pt'
        (path, name) = os.path.split(savename)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), savename)

        print(f'Epoch {epoch}: loss={epoch_loss:.4f}, cer={epoch_cer:.4f}, wer={epoch_wer:.4f}')

    print('Training complete.')

def combinationLoss(model, net):
    now = datetime.datetime.now()
    nowstr = now.strftime("%Y-%m-%d_%H-%M-%S")  # ファイル名に日付と時間を組み込む
    
    dataset = FrontProfile(
        opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train'
    )
    
    loader = dataset2dataloader(dataset)
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.base_lr,
        weight_decay=0.0,
        amsgrad=True
    )
    
    print(f'num_train_data: {len(dataset.data)}')
    crit = nn.CTCLoss()
    tic = time.time()

    train_logs = {'loss': [], 'cer': [], 'wer': []}
    csvname = f'logs/LipNet_{nowstr}.csv'

    # CSVヘッダーの書き込み
    with open(csvname, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['epoch', 'loss', 'wer', 'cer'])


    val_epoch = 0

    for epoch in range(opt.max_epoch):
        loss_sum, cer_sum, wer_sum = 0, 0, 0
        train_wer, train_cer = [], []

        for i_iter, input in enumerate(loader):
            model.train()

            vids = input.get('vids')
            vid1, vid2 = vids[0].cuda(), vids[1].cuda()
            txt = input.get('txt').cuda()
            tg = textgrid2dic(input.get('phonem'))

            optimizer.zero_grad()
            y1 = net(vid1)#横顔
            y2 = net(vid2)#正面
            

            # 損失の計算
            contrastive_loss = contrastive_phonem_loss(y1, y2, tg)

            ctc_crit = nn.CTCLoss(blank=0)  # CTC の blank インデックスを 0 と仮定
            ctc_loss = ctc_crit(
                y1.transpose(0, 1).log_softmax(-1),  # log_softmax を適用
                txt,
                input.get('vid_len').cuda().view(-1),
                input.get('txt_len').cuda().view(-1)
            )

            # 正規化された損失を組み合わせる
            max_contrastive_loss = 7.0713  # 事前に決定した最大値
            max_ctc_loss = 6.375          # 事前に決定した最大値
            alpha, beta = 0.5, 0.5        # 重みの調整パラメータ

            normalized_contrastive_loss = contrastive_loss / max_contrastive_loss
            normalized_ctc_loss = ctc_loss / max_ctc_loss
            loss = alpha * normalized_contrastive_loss + beta * normalized_ctc_loss

            # ロスの合計を記録
            loss_sum += loss.item()

            # 逆伝播
            loss.backward()
            if opt.is_optimize:
                optimizer.step()
            # デコードと評価
            pred_txt = ctc_decode(y1)
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))

            # ログ
            tot_iter = i_iter + epoch * len(loader)
            if tot_iter % opt.display == 0:
                avg_wer = np.mean(train_wer)
                avg_cer = np.mean(train_cer)
                writer.add_scalar('train loss', loss.item(), tot_iter)
                writer.add_scalar('train wer', avg_wer, tot_iter)
                writer.add_scalar('train cer', avg_cer, tot_iter)
                print('-' * 101)
                print(f'epoch={epoch}, tot_iter={tot_iter}, loss={loss.item()}, train_wer={avg_wer}')
                print('-' * 101)
            
            if tot_iter % opt.test_step == 0:
                val_loss, val_wer, val_cer = test(model, net, contrastive_phonem_loss)
                wandb.log({
                    "val_loss": val_loss,
                    "val_cer": val_cer,
                    "val_wer": val_wer
                }, step=val_epoch)
                val_epoch += 1
                writer.add_scalar('val loss', val_loss, tot_iter)
                writer.add_scalar('val wer', val_wer, tot_iter)
                writer.add_scalar('val cer', val_cer, tot_iter)

        # エポック終了後の統計
        epoch_loss = loss_sum / len(loader)
        epoch_cer = np.mean(train_cer)
        epoch_wer = np.mean(train_wer)
        train_logs['loss'].append(epoch_loss)
        train_logs['cer'].append(epoch_cer)
        train_logs['wer'].append(epoch_wer)

        # エポック単位のログを記録
        # エポック終了後の訓練ログ
        wandb.log({
            "train_loss": epoch_loss,
            "train_cer": epoch_cer,
            "train_wer": epoch_wer
        }, step=epoch)

        with open(csvname, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([epoch, epoch_loss, epoch_wer, epoch_cer])

        # モデル保存
        savename = f'{nowstr}_{opt.save_prefix}_loss_{epoch_loss:.4f}_wer_{epoch_wer:.4f}_cer_{epoch_cer:.4f}.pt'
        (path, name) = os.path.split(savename)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), savename)

        print(f'Epoch {epoch}: loss={epoch_loss:.4f}, cer={epoch_cer:.4f}, wer={epoch_wer:.4f}')

    print('Training complete.')





def train(model, net):

    now = datetime.datetime.now()
    # ファイル名に日付と時間を組み込む
    nowstr = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    dataset = FrontProfile(opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train')
        
    loader = dataset2dataloader(dataset) 
    optimizer = optim.Adam(model.parameters(),
                lr = opt.base_lr,
                weight_decay = 0.,
                amsgrad = True)
                
    print('num_train_data:{}'.format(len(dataset.data)))    
    crit = nn.CTCLoss()
    tic = time.time()
    
    train_wer = []
    train_cer = []
    
    for epoch in range(opt.max_epoch):
        loss_sum = 0
        cer_sum = 0
        wer_sum = 0
        for (i_iter, input) in enumerate(loader):
            model.train()
            
            vids = input.get('vids')
            vid1 , _  = vids
            vid1 = vid1.cuda()
            #vid2 = vid2.cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            
            optimizer.zero_grad()
            
            y = net(vid1) #[vid_len, batch , gru 512]
            #y2 = net(vid2)



            
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
            loss_sum += loss.item()
            loss.backward()
            if(opt.is_optimize):
                optimizer.step()

            
            
            tot_iter = i_iter + epoch*len(loader)
            
            pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))

            cer_sum += np.array(train_cer).mean()
            wer_sum += np.array(train_wer).mean()
            
            
            

            
            
            if(tot_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(loader)-i_iter)*v/3600.0
                
                writer.add_scalar('train loss', loss, tot_iter)
                writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)              
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))                
                print(''.join(101*'-'))
                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101*'-'))                
                print('epoch={},tot_iter={},eta={},loss={},train_wer={}'.format(epoch, tot_iter, eta, loss, np.array(train_wer).mean()))
                print(''.join(101*'-'))
                
            if(tot_iter % opt.test_step == 0):                
                (loss, wer, cer) = test(model, net,"")
                wandb.log({
                    "val_loss" : loss,
                    "val_cer" : cer,
                    "val_wer" : wer
                })
                print('i_iter={},lr={},loss={},wer={},cer={}'
                    .format(tot_iter,show_lr(optimizer),loss,wer,cer))
                writer.add_scalar('val loss', loss, tot_iter)                    
                writer.add_scalar('wer', wer, tot_iter)
                writer.add_scalar('cer', cer, tot_iter)
                savename = '{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.save_prefix, loss, wer, cer)
                (path, name) = os.path.split(savename)
                if(not os.path.exists(path)): os.makedirs(path)
                #torch.save(model.state_dict(), savename)
                if(not opt.is_optimize):
                    exit()
        epoch_loss = loss_sum / len(loader)  # 1epochのlossを計算
        epoch_cer = cer_sum / len(loader)
        epoch_wer = wer_sum / len(loader)

        wandb.log({
            "epoch" : epoch,
            "train_loss" : epoch_loss,
            "train_cer" : epoch_cer,
            "train_wer" : epoch_wer
        })

        
        savename = '{}_{}_loss_{}_wer_{}_cer_{}.pt'.format(nowstr,opt.save_prefix, epoch_loss, epoch_wer, epoch_cer)
        (path, name) = os.path.split(savename)
        if(not os.path.exists(path)): os.makedirs(path)
        #動作テスト中は消しておきます
        torch.save(model.state_dict(), savename)

        print('loss : {} cer: {}'.format(epoch_loss,epoch_cer))
        train_logs['loss'].append(epoch_loss)
        train_logs['cer'].append(epoch_cer)
        train_logs['wer'].append(epoch_wer)
    
   
    csvname = f'logs/LipNet_{nowstr}.csv'
    with open(csvname, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # ヘッダーを書き込む
        csvwriter.writerow(['epoch', 'loss', 'wer', 'cer'])
        # 各エポックのデータを書き込む
        for epoch in range(len(train_logs['loss'])):
            loss = train_logs['loss'][epoch]
            wer = train_logs['wer'][epoch]
            cer = train_logs['cer'][epoch]
            csvwriter.writerow([epoch, loss, wer, cer])


#valのlossを記録できるように変更
def train2(model, net):
    now = datetime.datetime.now()
    nowstr = now.strftime("%Y-%m-%d_%H-%M-%S")  # ファイル名に日付と時間を組み込む
    
    dataset = FrontProfile(
        opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train'
    )
    
    loader = dataset2dataloader(dataset)
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.base_lr,
        weight_decay=0.0,
        amsgrad=True
    )
    
    print(f'num_train_data: {len(dataset.data)}')
    crit = nn.CTCLoss()
    tic = time.time()

    train_logs = {'loss': [], 'cer': [], 'wer': []}
    csvname = f'logs/LipNet_{nowstr}.csv'

    # CSVヘッダーの書き込み
    with open(csvname, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['epoch', 'loss', 'wer', 'cer'])


    val_epoch = 0

    for epoch in range(opt.max_epoch):
        loss_sum, cer_sum, wer_sum = 0, 0, 0
        train_wer, train_cer = [], []

        for i_iter, input in enumerate(loader):
            model.train()

            vids = input.get('vids')
            vid1 = vids[0].cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()

            optimizer.zero_grad()
            y = net(vid1)

            # CTC Loss の計算
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
            loss_sum += loss.item()
            loss.backward()
            if opt.is_optimize:
                optimizer.step()

            # デコードと評価
            pred_txt = ctc_decode(y)
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))

            # イテレーションごとのログ
            tot_iter = i_iter + epoch * len(loader)
            if tot_iter % opt.display == 0:
                avg_wer = np.mean(train_wer)
                avg_cer = np.mean(train_cer)
                writer.add_scalar('train loss', loss.item(), tot_iter)
                writer.add_scalar('train wer', avg_wer, tot_iter)
                print('-' * 101)
                print(f'epoch={epoch}, tot_iter={tot_iter}, loss={loss.item():.4f}, train_wer={avg_wer:.4f}')
                print('-' * 101)
            
            # 検証
            if tot_iter % opt.test_step == 0:
                val_loss, val_wer, val_cer = test(model, net, "")
            
                wandb.log({
                    "val_loss": val_loss,
                    "val_cer": val_cer,
                    "val_wer": val_wer
                }, step=val_epoch)
                val_epoch += 1
                writer.add_scalar('val loss', val_loss, tot_iter)
                writer.add_scalar('val wer', val_wer, tot_iter)
                writer.add_scalar('val cer', val_cer, tot_iter)

        # エポック終了後の統計
        epoch_loss = loss_sum / len(loader)
        epoch_cer = np.mean(train_cer)
        epoch_wer = np.mean(train_wer)
        train_logs['loss'].append(epoch_loss)
        train_logs['cer'].append(epoch_cer)
        train_logs['wer'].append(epoch_wer)

        # エポック単位のログを記録
        wandb.log({
            "train_loss": epoch_loss,
            "train_cer": epoch_cer,
            "train_wer": epoch_wer
        }, step=epoch)

        with open(csvname, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([epoch, epoch_loss, epoch_wer, epoch_cer])

        # モデル保存
        savename = f'{nowstr}_{opt.save_prefix}_loss_{epoch_loss:.4f}_wer_{epoch_wer:.4f}_cer_{epoch_cer:.4f}.pt'
        (path, name) = os.path.split(savename)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), savename)

        print(f'Epoch {epoch}: loss={epoch_loss:.4f}, cer={epoch_cer:.4f}, wer={epoch_wer:.4f}')

    print('Training complete.')




                
if(__name__ == '__main__'):
    print("Loading options...")
    model = LipNet()
    model = model.cuda()
    net = nn.DataParallel(model).cuda()

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


    if hasattr(opt, 'weights'):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()

        # ロード可能なパラメータをフィルタリング
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if k not in pretrained_dict.keys()]

        # ロード成功・失敗の状況を出力
        if len(pretrained_dict) > 0:
            print(f"[INFO] ロード成功: {len(pretrained_dict)} / {len(model_dict)} パラメータが一致しました。")
            print(f"[INFO] ロードされたパラメータ: {list(pretrained_dict.keys())}")
        else:
            print("[WARNING] ロード可能なパラメータがありませんでした。モデルの構造が異なる可能性があります。")

        if len(missed_params) > 0:
            print(f"[WARNING] ロード失敗: {len(missed_params)} パラメータが一致しませんでした。")
            print(f"[WARNING] 一致しなかったパラメータ: {missed_params}")

        # パラメータを更新してモデルにロード
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    else:
        print("[INFO] 'opt.weights'が指定されていません。事前学習済みモデルはロードされません。")

        
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    #maping(model, net)
    #pre_train2(model, net)
    #train(model,net)
    #combinationLoss(model,net)
    train2(model,net)
        
