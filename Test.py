import os
import openpyxl

from Model import *
from Covid19DataSet import *
class Test:

    def __init__(self, input_dim, test_dataloader, load_model_name):
        self.input_dim=input_dim
        self.test_dataloader=test_dataloader
        self.model_name=load_model_name
    def load_model(self):
        self.model=Model(self.input_dim)
        self.model.load_state_dict(torch.load(self.model_name+".pth"))
    def Test(self,save_csv_name):
        self.load_model()
        pred_list = []
        for data in self.test_dataloader:
            preds = self.model(data)
            for pred in preds:
                pred_list.append(float(pred))
        #os.chdir("..")
        print(os.getcwd())
        work_book = openpyxl.Workbook()
        work_sheet = work_book.create_sheet('Submission')
        row = 1
        column = 1
        for id, tested_positive in enumerate(pred_list):
            work_sheet.cell(row, column).value = id
            column += 1
            work_sheet.cell(row, column).value = tested_positive
            row += 1
            column = 1
        del work_book["Sheet"]
        work_book.save(save_csv_name+'.csv')

