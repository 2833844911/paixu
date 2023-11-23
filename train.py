import json
import random

from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import re
from tqdm import tqdm
import torch

if torch.cuda.is_available():
    x = "cuda"
else:
    x = "cpu"

device = torch.device(x)

allkey = 0
torch.autograd.set_detect_anomaly(True)
class classInfo(Dataset):
    def __init__(self, wz=4):
        global allkey, kooo2

        alltxt = os.listdir('./file')
        wb = ''
        if qidqyxx == 1:
            self.dt = json.load(open('./ci.json', encoding='utf-8'))
        else:
            self.dt = {'ks':3}
        for i in alltxt:
            with open('./file/' + i, encoding='utf-8') as f:
                wb += f.read()
        self.allcan = []
        wb = re.sub(r'[^\u4e00-\u9fff]', wb, ' ').replace('。',' ').replace('、',' ').replace('；',' ').replace('.',' ').replace('-',' ').replace('\n',' ').replace('”',' ').replace('，',' ').replace('！',' ').replace('“',' ')
        with open('./a.txt', 'w',encoding='utf-8') as f:
            f.write(wb)

        for i in list(wb):
            if i in self.dt:
                continue
            else:
                self.dt[i] = self.dt['ks']
                self.dt['ks'] += 1
        kb = tqdm(range(len(wb)-wz*2))
        kb.set_description(desc="准备数据集中")
        for i in kb:
            for jj in range(wz):
                key = wb[i:i+jj]
                if len(key) >3 and self.contains_non_chinese(key[0]) == True:

                    if self.contains_non_chinese(key[-1]) == True:
                        if self.contains_non_chinese(key[1:-1]) == True:
                            break

                    elif jj == wz-1:
                        if self.contains_non_chinese(key[1:-1]) == True:
                            break
                    else:
                        continue
                    k = [self.dt[r] for r in list(key[1:-1])]
                    lp = k.copy()

                    l = [0] * (wz - len(k))
                    random.shuffle(k)

                    kooo = [1] + k
                    kooo = kooo + [2]

                    kooo2 = [1] + lp
                    kooo2 = kooo2 + [2]
                    kooo2 += l

                    kooo = kooo + l
                    if len(kooo) != 22:
                        print('====>')
                    self.allcan.append([kooo,kooo2])


        allkey = len(self.dt) + 5
        json.dump(self.dt, open('./ci.json','w', encoding='utf-8'), ensure_ascii=False)

        self.allcan = self.allcan[::-1]
        self.allcan = self.allcan[batchsize:]
        self.length = len(self.allcan)

    def contains_non_chinese(self, text):
        return bool(re.search(r'[^\u4e00-\u9fff]', text))

    def __getitem__(self, item):
        return torch.tensor(self.allcan[item][0]),torch.tensor(self.allcan[item][1])
    def __len__(self):
        return self.length

hidden_size = 150


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.eig = nn.Embedding(allkey, hidden_size)
        self.engru = nn.GRU(hidden_size, 256,batch_first=True, num_layers=2,bidirectional=True, dropout=0.5)

        self.Linetowz = nn.Linear(2*256, allkey)
        self.oooooooooo = nn.Linear(150, 512)
        self.sf = nn.Softmax(dim=1)

        self.conv1d = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=1, stride=1)

        self.degru = nn.GRU(hidden_size, 256,batch_first=True, num_layers=2, dropout=0.5)

        self.linQ = nn.Linear(512, 150)
        self.linK = nn.Linear(512, 150)
        self.linV = nn.Linear(512, 150)


    def forward(self,x,target):
        dt = self.eig(x)
        jjj,endt = self.engru(dt)
        endt = endt.permute(1, 0, 2)
        endte2 =self.conv1d( endt).permute(1, 0, 2)
        endte2 = endte2.contiguous()
        itinfo = torch.zeros(x.shape[0], 1, hidden_size).to(device)
        output = torch.zeros(x.shape[0], 1, allkey).to(device)
        x_modified = x.clone()

        for idx in range(x.shape[1]):

            koooo = self.oooooooooo(itinfo)
            kkkkk = self.linK(koooo)
            vvvvv = self.linV(koooo)


            ttttt = self.linQ(koooo)
            ttttt = ttttt.expand(-1,kkkkk.shape[1],-1)
            gx = torch.bmm(ttttt, torch.transpose(kkkkk, 1,2))
            gx = self.sf(gx)
            data = torch.bmm(gx, vvvvv )


            cee,endte2 = self.degru(data, endte2)
            endt2 = endte2.permute(1, 0, 2)
            endt2 = endt2.reshape(endt2.shape[0],-1)
            scinfo = self.Linetowz(endt2)
            selected_values = torch.zeros(x.shape[0], x.shape[1]).to(device)
            for i in range(x.shape[0]):
                wz = x_modified[i] == -1
                x_modified[i][wz] = 0
                selected_values[i] = scinfo[i, x_modified[i]]
                x_modified[i][wz] = -1
                selected_values[i][wz] = float('-inf')
            max_indices = torch.max(selected_values, dim=1).indices.reshape(x.shape[0], 1)
            scinfo = scinfo.reshape(x.shape[0], 1, allkey)
            output = torch.concat([output, scinfo], dim=1)

            selected_values2 = torch.zeros(x.shape[0], 1, dtype=torch.int).to(device)
            for i in range(x.shape[0]):
                selected_values2[i,0] = x[i, max_indices[i,0]]
                x_modified[i, max_indices[i, 0]] = -1
            if random.randint(0, 100) > 50:
                emdt = self.eig(selected_values2)
            else:
                emdt = self.eig(target[:, idx].reshape(x.shape[0], 1))
            itinfo = torch.concat([itinfo, emdt],dim=1)

        return output[:, 1:]







if __name__ == '__main__':
    batchsize = 20
    qidqyxx = 0 #开启迁移学习

    dataLoad = classInfo(20)


    sjmode = Model()
    if qidqyxx == 1:
        sjmode.load_state_dict(torch.load("./model_epoch.pth"))


    sjmode.to(device)
    best_loss = float('inf')
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(sjmode.parameters(),lr=0.0005)
    for _ in range(100):
        dataAll = DataLoader(dataLoad, batch_size=batchsize, shuffle=True)
        sjmode.train()
        lossAll = 0
        cs = 0
        fh = tqdm(dataAll)
        for info,target in fh:
            info = info.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            out = sjmode(info, target)
            oute = out.reshape(-1, out.shape[-1])
            trg = target.reshape(-1)
            loss = criterion(oute,trg)
            loss.backward()
            optimizer.step()
            lossAll += loss.item()
            cs += 1
            fh.set_description(desc="epoch {} loss {}".format(_,lossAll/cs ))
         # 检查是否是最好的模型
        if lossAll / cs < best_loss:
            best_loss = lossAll / cs
            # 保存模型
            torch.save(sjmode.state_dict(), f'model_epoch.pth')

            print(f"\nModel saved: epoch {_}, loss {best_loss:.4f}")



