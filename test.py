import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import re
import json
if torch.cuda.is_available():
    x = "cuda"
else:
    x = "cpu"

device = torch.device(x)
dt = json.load(open('./ci.json', encoding='utf-8'))
fdt = {}
for k,v in dt.items():
    fdt[str(v)] = k

allkey = len(dt)+5


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

        self.linQ = nn.Linear(256*2, 150)
        self.linK = nn.Linear(512, 150)
        self.linV = nn.Linear(512, 150)


    def forward(self,x):
        dt = self.eig(x)
        jjj, endt = self.engru(dt)
        endt = endt.permute(1, 0, 2)
        endte2 =self.conv1d( endt).permute(1, 0, 2)
        endte2 = endte2.contiguous()
        itinfo = torch.zeros(x.shape[0], 1, hidden_size).to(device)
        output = torch.zeros(x.shape[0], 1, allkey).to(device)
        output2 = torch.zeros(x.shape[0], 1).to(device)
        x_modified = x.clone()


        for idx in range(x.shape[1]):
            koooo = self.oooooooooo(itinfo)
            kkkkk = self.linK(koooo)
            vvvvv = self.linV(koooo)

            ttttt = self.linQ(koooo)
            # ttttt = self.linQ(endte2.permute(1, 0, 2).reshape(x.shape[0], 1, -1))
            ttttt = ttttt.expand(-1,kkkkk.shape[1],-1)
            # ll = ttttt.permute(0, 2, 1)
            gx = torch.bmm(ttttt, torch.transpose(kkkkk, 1,2))
            gx = self.sf(gx)
            data = torch.bmm(gx, vvvvv )

            cee, endte2 = self.degru(data, endte2)

            # cee,endte2 = self.degru(itinfo, endte2)
            endt2 = endte2.permute(1, 0, 2)
            endt2 = endt2.reshape(endt2.shape[0],-1)
            scinfo = self.Linetowz(endt2)
            # scinfo = endt2
            selected_values = torch.zeros(x.shape[0], x.shape[1]).to(device)
            for i in range(x.shape[0]):
                wz = x_modified[i]==-1
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
            output2 = torch.concat([output2, selected_values2],dim=1)

            emdt = self.eig(selected_values2)
            itinfo = torch.concat([itinfo, emdt],dim=1)

        return output2[:, 1:]

dymodel = Model()
dymodel.load_state_dict(torch.load("./model_epoch.pth",map_location=device))
dymodel.to(device)
dymodel.eval()
def getinfo(text):
    me = 20
    aldtoo = [0]*(me - len(text))

    aldt = [1]
    for i in list(text):
        aldt.append(dt[i])
    aldt += [2]
    aldt += aldtoo
    at = torch.tensor(aldt).to(device)
    at = at.reshape([1,-1])
    ot = dymodel(at)
    ot = ot.tolist()
    jg = ''
    for i in ot[0]:
        if i <= 2:
            continue
        jg += fdt[str(int(i))]
    return jg






if __name__ == '__main__':
    px = getinfo("们学校晃荡那我先去一圈")
    print(px)
