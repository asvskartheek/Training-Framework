import torch 
import torch.nn as nn

class BinaryClassification(nn.Module):
    def __init__(self,input_feats,hidden_size):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(input_feats, hidden_size) 
        self.layer_out = nn.Linear(hidden_size, 1) 
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.layer_out(x)
        x = self.sigmoid(x)
        
        return x