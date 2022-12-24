from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split

from Train_Valid_Optimize import *
from Test import *
from  Correlation import *

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
config={
    'random_state':20,
    'train_size':0.8,
    'batch_size':512,
    'learning_rate':0.001,
    'epoches':3000,
    'model_name':"covid_model2"
}

train_df=pd.read_csv('D:/PythonProjects/MachineLeaning/CovidPredict_ML1/Data/covid_train.csv')
test_df=pd.read_csv('D:/PythonProjects/MachineLeaning/CovidPredict_ML1/Data/covid_test.csv')

print(train_df.shape[1])

Corr=Correlation(train_df,test_df,result_col_num=94)

train_df,test_df=Corr.cal_corr(0.5)

x=train_df[train_df.columns[1:train_df.shape[1]-1]]
y=train_df[train_df.columns[train_df.shape[1]-1]]
z=test_df[test_df.columns[1:train_df.shape[1]-1]]





x_train,x_valid,y_train,y_valid=train_test_split(x,y,train_size=config['train_size'],random_state=config['random_state'])

train_dataset=Covid19DataSet(x_train,y_train)
valid_dataset=Covid19DataSet(x_valid,y_valid)
test_dataset=Covid19DataSet(z)

train_DataLoader=DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True)
valid_DataLoader=DataLoader(valid_dataset,batch_size=config['batch_size'],shuffle=True)
test_DataLoader=DataLoader(test_dataset,batch_size=config['batch_size'],shuffle=False)

Train_Valid_Optimize=Train_Valid_Optimize(x_train.shape[1],train_DataLoader,valid_DataLoader,config["epoches"],config["learning_rate"],device,config["model_name"])
Train_Valid_Optimize.train_valid_optimize()
Test=Test(z.shape[1],test_DataLoader,"covid_model2")
Test.Test("test")

