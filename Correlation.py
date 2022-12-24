import numpy as np
import  pandas as pd

class Correlation:

    def __init__(self,train_df,test_df,result_col_num):
        self.train_df=train_df
        self.test_df=test_df
        self.result_col_num=result_col_num
    def cal_corr(self,threshold):
        remove_col_list=[]
        result_df= self.train_df.iloc[:, [self.result_col_num]]
        result_series=result_df[result_df.columns[0]]
        for col_index,column in enumerate(self.train_df.columns, start=0):
            if (column!=self.train_df.columns[94])&(column != self.train_df.columns[0]):
                feature_df= self.train_df.loc[:, [column]]
                feature_series=feature_df[feature_df.columns[0]]
                corr=np.corrcoef(feature_series,result_series)[0,1]
                if(abs(corr)<threshold):
                    remove_col_list.append(column)
        for column in remove_col_list:
            self.train_df=self.train_df.drop(column, axis=1)
            self.test_df=self.test_df.drop(column,axis=1)
        return self.train_df, self.test_df



