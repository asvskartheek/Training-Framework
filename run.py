import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *
from model import BinaryClassification

from fastai.basics import *

from pathlib import Path

# Pre-Processing & Databunch generation
df = pd.read_csv('Dataset_spine.csv')
df.drop('Unnamed: 13',axis=1,inplace=True)
df['Class_att'] = df['Class_att'].astype('category')
encode_map = {
		'Abnormal': 1,
		'Normal': 0
}
df['Class_att'].replace(encode_map, inplace=True)

X = np.array(df.iloc[:, 0:-1]).astype(np.float64)
y = np.array(df.iloc[:, -1]).astype(np.float64)
data_bunch = generate_databunch(X,y)

# Base Model
base_model = BinaryClassification(X.shape[1],64) # feats, hidden_size
learner = Learner(data=data_bunch, model=base_model.double(), loss_func=nn.BCELoss())

# Find Ideal LR
learner.lr_find()
lr_finder = learner.recorder.plot(return_fig=True)
plt.show(lr_finder)
Path("./viz/").mkdir(parents=True, exist_ok=True)
lr_finder.savefig('viz/lr_finder.png')

# User Input
n_epochs = int(input('Number of Epochs?'))
lr = float(input('Learning Rate?'))

# Train
learner.fit_one_cycle(n_epochs, lr)
learner.save('model')

# Finish
training_proc = learner.recorder.plot_losses(return_fig=True)
training_proc.savefig('viz/training_proc.png')

print('Training History plotted at viz/training_proc.png')
print('Trained model saved at models/model.pth')