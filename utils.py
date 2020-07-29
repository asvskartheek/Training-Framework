from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from fastai.basics import DataBunch

class TabularData(Dataset):
		def __init__(self, X, y):
						self.X = X
						self.y = y

		def __len__(self):
				return self.X.shape[0]

		def __getitem__(self, idx):
				return self.X[idx, :], self.y[idx]

def train_valid_test(X,y,test,valid):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=69)
		X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid, random_state=69)

		return X_train,X_valid,X_test,y_train,y_valid,y_test

def generate_databunch(X,y,test=0.3,valid=0.1,batch_size=32):
		X_train,X_valid,X_test,y_train,y_valid,y_test = train_valid_test(X,y,test,valid)
		data_bunch = DataBunch.create(
				train_ds=TabularData(X_train,y_train)
				,valid_ds=TabularData(X_valid, y_valid)
				,test_ds = TabularData(X_test, y_test), bs=batch_size)

		return data_bunch