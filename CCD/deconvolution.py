import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.utils.data as Data

class D1(nn.Module):
    def __init__(self, gene_size, unit):
        super(D1, self).__init__()
        self.gene_size = gene_size
        self.unit = unit
        self.D1 = nn.Sequential(
            nn.Linear(self.gene_size, 1024),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(64, self.unit),
            nn.Softmax(dim=1),
        )
        
       
    def forward(self, inputs):
        out = self.D1(inputs)
        return out

class D2(nn.Module):
    def __init__(self, gene_size, unit):
        super(D2, self).__init__()
        self.gene_size = gene_size
        self.unit = unit
        
        self.D2 = nn.Sequential(
            nn.Linear(self.gene_size, 512),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(64, self.unit),
            nn.Softmax(dim=1),
        )
        
     
    def forward(self, inputs):
        out = self.D2(inputs)
        return out
    
class D3(nn.Module):
    def __init__(self, gene_size, unit):
        super(D3, self).__init__()
        self.gene_size = gene_size
        self.unit = unit
        
        self.D3 = nn.Sequential(
            nn.Linear(self.gene_size, 256),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(64, self.unit),
            nn.Softmax(dim=1),
        )
        
     
    def forward(self, inputs):
        out = self.D3(inputs)
        return out

D1, D2, D3 = D1(1971,12), D2(1971,12), D3(1971,12)
D1.load_state_dict(torch.load('./CCD/weight/model1.pt', weights_only=True))
D2.load_state_dict(torch.load('./CCD/weight/model2.pt', weights_only=True))
D3.load_state_dict(torch.load('./CCD/weight/model3.pt', weights_only=True))
D1.eval()
D2.eval()
D3.eval()

lihc = pd.read_csv("lihc_data.csv",index_col=0)
test = torch.tensor(np.log10(lihc.to_numpy()+1), dtype=torch.float64)
test_label = torch.zeros([lihc.to_numpy().shape[0],12])
test_dataset = Data.TensorDataset(test, test_label)
test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=False)
test_est = []
with torch.no_grad():
    for n, D in enumerate(test_dataset):
        pred1 = D1(D[0].float())
        pred2 = D2(D[0].float())
        pred3 = D3(D[0].float())
        for m in range(len(D[1])):
            pred = (pred1.cpu().detach().numpy()[m] + pred2.cpu().detach().numpy()[m] + pred3.cpu().detach().numpy()[m])/3
            test_est.append(pred)
pd.DataFrame(test_est, columns=celltype.columns).to_csv("./lihc_fraction/lihc_fraction.csv")
