import pandas as pd
import numpy as np
import multiprocessing
import warnings
warnings.filterwarnings('ignore')
pd.set_option('mode.chained_assignment', None)

# 首先执行下面的全部函数
class feature_generation(object):

    def __init__(self, data, feature_list, p_list, core_num=1):
        self.data = data  # 包含基础变量的数据
        self.feature_list = feature_list  # 变量名前缀
        self.p_list = p_list  # 变量名前缀
        self.df = pd.DataFrame([])  # 用于收集最终变量的数据框
        self.func_list = ['Num', 'Nmz', 'Evr', 'Avg', 'Tot', 't2T', 'Max', 'Min', 'Msg', 'Msz',
                          'Cav', 'Cmn', 'Std', 'Cva', 'Cmm', 'Cnm', 'Cxm', 'Cxp', 'Ran', 'Nci', 'Ncd',
                          'Ncn', 'Pdn', 'Cmx', 'Cmp', 'Cnp', 'Msx', 'Trm', 'Bup', 'Mai', 'Mad',
                          'Rpp', 'Dpp', 'Mpp', 'Npp']
        self.core_num = core_num  # 35个函数对应35个核

    def fit_generate(self):
        """
        通过循环变量名inv和月份p，
        实现全部变量的衍生
        """
        for self.inv in self.feature_list:
            for self.p in self.p_list:
                var_df = self.generate(self.inv, self.p)
                self.df = pd.concat([self.df, var_df], axis=1)
        return self.df

    def generate(self, inv, p):
        """
        多进程，衍生变量主函数
        """
        var_df = pd.DataFrame([])
        pool = multiprocessing.Pool(self.core_num)
        results = [pool.apply_async(self.auto_var, [func]) for func in self.func_list]
        pool.close()
        pool.join()
        for i in range(len(results)):
            try:
                columns, value = results[i].get()
                var_df[columns] = value
            except:
                continue
        return var_df

    # 定义批量调用双参数的函数
    def auto_var(self, func):
        if func == 'Num':
            try:
                return self.Num(self.inv, self.p)
            except:
                print("Num PARSE ERROR", self.inv, self.p)
        elif func == 'Nmz':
            try:
                return self.Nmz(self.inv, self.p)
            except:
                print("Nmz PARSE ERROR", self.inv, self.p)

        elif func == 'Evr':
            try:
                return self.Evr(self.inv, self.p)
            except:
                print("Evr PARSE ERROR", self.inv, self.p)
        elif func == 'Avg':
            try:
                return self.Avg(self.inv, self.p)
            except:
                print("Avg PARSE ERROR", self.inv, self.p)
        elif func == 'Tot':
            try:
                return self.Tot(self.inv, self.p)
            except:
                print("Tot PARSE ERROR", self.inv, self.p)
        elif func == 'Tot2T':
            try:
                return self.Tot2T(self.inv, self.p)
            except:
                print("Tot2T PARSE ERROR", self.inv, self.p)
        elif func == 'Max':
            try:
                return self.Max(self.inv, self.p)
            except:
                print("Tot PARSE ERROR", self.inv, self.p)
        elif func == 'Min':
            try:
                return self.Min(self.inv, self.p)
            except:
                print("Min PARSE ERROR", self.inv, self.p)
        elif func == 'Msg':
            try:
                return self.Msg(self.inv, self.p)
            except:
                print("Msg PARSE ERROR", self.inv, self.p)
        elif func == 'Msz':
            try:
                return self.Msz(self.inv, self.p)
            except:
                print("Msz PARSE ERROR", self.inv, self.p)
        elif func == 'Cav':
            try:
                return self.Cav(self.inv, self.p)
            except:
                print("Cav PARSE ERROR", self.inv, self.p)
        elif func == 'Cmn':
            try:
                return self.Cmn(self.inv, self.p)
            except:
                print("Cmn PARSE ERROR", self.inv, self.p)
        elif func == 'Std':
            try:
                return self.Std(self.inv, self.p)
            except:
                print("Std PARSE ERROR", self.inv, self.p)
        elif func == 'Cva':
            try:
                return self.Cva(self.inv, self.p)
            except:
                print("Cva PARSE ERROR", self.inv, self.p)
        elif func == 'Cmm':
            try:
                return self.Cmm(self.inv, self.p)
            except:
                print("Cmm PARSE ERROR", self.inv, self.p)
        elif func == 'Cnm':
            try:
                return self.Cnm(self.inv, self.p)
            except:
                print("Cnm PARSE ERROR", self.inv, self.p)
        elif func == 'Cxm':
            try:
                return self.Cxm(self.inv, self.p)
            except:
                print("Cxm PARSE ERROR", self.inv, self.p)
        elif func == 'Cxp':
            try:
                return self.Cxp(self.inv, self.p)
            except:
                print("Cxp PARSE ERROR", self.inv, self.p)
        elif func == 'Ran':
            try:
                return self.Ran(self.inv, self.p)
            except:
                print("Ran PARSE ERROR", self.inv, self.p)
        elif func == 'Nci':
            try:
                return self.Nci(self.inv, self.p)
            except:
                print("Nci PARSE ERROR", self.inv, self.p)
        elif func == 'Pdn':
            try:
                return self.Pdn(self.inv, self.p)
            except:
                print("Pdn PARSE ERROR", self.inv, self.p)
        elif func == 'Cmx':
            try:
                return self.Cmx(self.inv, self.p)
            except:
                print("Cmx PARSE ERROR", self.inv, self.p)
        elif func == 'Cmp':
            try:
                return self.Cmp(self.inv, self.p)
            except:
                print("Cmp PARSE ERROR", self.inv, self.p)
        elif func == 'Cnp':
            try:
                return self.Cnp(self.inv, self.p)
            except:
                print("Cnp PARSE ERROR", self.inv, self.p)
        elif func == 'Msx':
            try:
                return self.Msx(self.inv, self.p)
            except:
                print("Msx PARSE ERROR", self.inv, self.p)
        elif func == 'Trm':
            try:
                return self.Trm(self.inv, self.p)
            except:
                print("Trm PARSE ERROR", self.inv, self.p)
        elif func == 'Bup':
            try:
                return self.Bup(self.inv, self.p)
            except:
                print("Bup PARSE ERROR", self.inv, self.p)
        elif func == 'Ncd':
            try:
                return self.Ncd(self.inv, self.p)
            except:
                print("Ncd PARSE ERROR", self.inv, self.p)
        elif func == 'Ncn':
            try:
                return self.Ncn(self.inv, self.p)
            except:
                print("Ncn PARSE ERROR", self.inv, self.p)
        elif func == 'Mai':
            try:
                return self.Mai(self.inv, self.p)
            except:
                print("Mai PARSE ERROR", self.inv, self.p)
        elif func == 'Mad':
            try:
                return self.Mad(self.inv, self.p)
            except:
                print("Mad PARSE ERROR", self.inv, self.p)
        elif func == 'Rpp':
            try:
                return self.Rpp(self.inv, self.p)
            except:
                print("Rpp PARSE ERROR", self.inv, self.p)
        elif func == 'Dpp':
            try:
                return self.Dpp(self.inv, self.p)
            except:
                print("Dpp PARSE ERROR", self.inv, self.p)
        elif func == 'Mpp':
            try:
                return self.Mpp(self.inv, self.p)
            except:
                print("Mpp PARSE ERROR", self.inv, self.p)
        elif func == 'Npp':
            try:
                return self.Npp(self.inv, self.p)
            except:
                print("Npp PARSE ERROR", self.inv, self.p)
    """
    35个衍生函数，inv为变量名，p为月份（时间切片）
    """

    def Num(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)].values
        auto_value = np.where(df > 0, 1, 0).sum(axis=1)
        return inv + '_num' + str(p), auto_value


    def Nmz(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)].values
        auto_value = np.where(df == 0, 1, 0).sum(axis=1)
        return inv + '_nmz' + str(p), auto_value


    def Evr(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)].values
        arr = np.where(df > 0, 1, 0).sum(axis=1)
        auto_value = np.where(arr, 1, 0)
        return inv + '_evr' + str(p), auto_value


    def Avg(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)].values
        auto_value = np.nanmean(df, axis=1)
        return inv + '_avg' + str(p), auto_value


    def Tot(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)].values
        auto_value = np.nansum(df, axis=1)
        return inv + '_tot' + str(p), auto_value


    def Tot2T(self, inv, p):
        df = self.data.loc[:, inv + '2':inv + str(p + 1)].values
        auto_value = df.sum(1)
        return inv + '_tot2t' + str(p), auto_value


    def Max(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)]
        auto_value = np.nanmax(df, axis=1)
        return inv + '_max' + str(p), auto_value


    def Min(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)].values
        auto_value = np.nanmin(df, axis=1)
        return inv + '_min' + str(p), auto_value


    def Msg(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)].values
        df_value = np.where(df > 0, 1, 0)
        auto_value = []
        for i in range(len(df_value)):
            row_value = df_value[i, :]
        if row_value.max() <= 0:
            indexs = '0'
            auto_value.append(indexs)
        else:
            indexs = 1
        for j in row_value:
            if j > 0:
                break
        indexs += 1
        auto_value.append(indexs)
        return inv + '_msg' + str(p), auto_value


    def Msz(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)].values
        df_value = np.where(df == 0, 1, 0)
        auto_value = []
        for i in range(len(df_value)):
            row_value = df_value[i, :]
        if row_value.max() <= 0:
            indexs = '0'
            auto_value.append(indexs)
        else:
            indexs = 1
        for j in row_value:
            if j > 0:
                break
        indexs += 1
        auto_value.append(indexs)
        return inv + '_msz' + str(p), auto_value


    def Cav(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)]
        auto_value = df[inv + '1'] / (np.nanmean(df, axis=1) + 1e-10)
        return inv + '_cav' + str(p), auto_value


    def Cmn(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)]
        auto_value = df[inv + '1'] / (np.nanmin(df, axis=1) + 1e-10)
        return inv + '_cmn' + str(p), auto_value


    def Mai(self, inv, p):
        arr = self.data.loc[:, inv + '1':inv + str(p)].values
        auto_value = []
        for i in range(len(arr)):
            df_value = arr[i, :]
        value_lst = []
        for k in range(len(df_value) - 1):
            minus = df_value[k] - df_value[k + 1]
        value_lst.append(minus)
        auto_value.append(np.nanmax(value_lst))
        return inv + '_mai' + str(p), auto_value


    def Mad(self, inv, p):
        arr = self.data.loc[:, inv + '1':inv + str(p)].values
        auto_value = []
        for i in range(len(arr)):
            df_value = arr[i, :]
        value_lst = []
        for k in range(len(df_value) - 1):
            minus = df_value[k + 1] - df_value[k]
        value_lst.append(minus)
        auto_value.append(np.nanmax(value_lst))
        return inv + '_mad' + str(p), auto_value


    def Std(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)].values
        auto_value = np.nanvar(df, axis=1)
        return inv + '_std' + str(p), auto_value


    def Cva(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)]
        auto_value = np.nanmean(df, axis=1) / (np.nanvar(df, axis=1) + 1e-10)
        return inv + '_cva' + str(p), auto_value


    def Cmm(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)]
        auto_value = df[inv + '1'] - np.nanmean(df, axis=1)
        return inv + '_cmm' + str(p), auto_value


    def Cnm(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)]
        auto_value = df[inv + '1'] - np.nanmin(df, axis=1)
        return inv + '_cnm' + str(p), auto_value


    def Cxm(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)]
        auto_value = df[inv + '1'] - np.nanmax(df, axis=1)
        return inv + '_cxm' + str(p), auto_value


    def Cxp(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)]
        temp = np.nanmin(df, axis=1)
        auto_value = (df[inv + '1'] - temp) / (temp + 1e-10)
        return inv + '_cxp' + str(p), auto_value


    def Ran(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)].values
        auto_value = np.nanmax(df, axis=1) - np.nanmin(df, axis=1)
        return inv + '_ran' + str(p), auto_value


    def Nci(self, inv, p):
        arr = self.data.loc[:, inv + '1':inv + str(p)].values


        auto_value = []
        for i in range(len(arr)):
            df_value = arr[i, :]
        value_lst = []
        for k in range(len(df_value) - 1):
            minus = df_value[k] - df_value[k + 1]
        value_lst.append(minus)
        value_ng = np.where(np.array(value_lst) > 0, 1, 0).sum()
        auto_value.append(np.nanmax(value_ng))
        return inv + '_nci' + str(p), auto_value


    def Ncd(self, inv, p):
        arr = self.data.loc[:, inv + '1':inv + str(p)].values
        auto_value = []
        for i in range(len(arr)):
            df_value = arr[i, :]
        value_lst = []
        for k in range(len(df_value) - 1):
            minus = df_value[k] - df_value[k + 1]
        value_lst.append(minus)
        value_ng = np.where(np.array(value_lst) < 0, 1, 0).sum()
        auto_value.append(np.nanmax(value_ng))
        return inv + '_ncd' + str(p), auto_value


    def Ncn(self, inv, p):
        arr = self.data.loc[:, inv + '1':inv + str(p)].values
        auto_value = []
        for i in range(len(arr)):
            df_value = arr[i, :]
        value_lst = []
        for k in range(len(df_value) - 1):
            minus = df_value[k] - df_value[k + 1]
        value_lst.append(minus)
        value_ng = np.where(np.array(value_lst) == 0, 1, 0).sum()
        auto_value.append(np.nanmax(value_ng))
        return inv + '_ncn' + str(p), auto_value


    def Bup(self, inv, p):
        arr = self.data.loc[:, inv + '1':inv + str(p)].values
        auto_value = []
        for i in range(len(arr)):
            df_value = arr[i, :]
        index = 0
        for k in range(len(df_value) - 1):
            if df_value[k] > df_value[k + 1]:
                break
        index = + 1
        if index == p:
            value = 1
        else:
            value = 0
        auto_value.append(value)
        return inv + '_bup' + str(p), auto_value


    def Pdn(self, inv, p):
        arr = self.data.loc[:, inv + '1':inv + str(p)].values
        auto_value = []
        for i in range(len(arr)):
            df_value = arr[i, :]
        index = 0
        for k in range(len(df_value) - 1):
            if df_value[k + 1] > df_value[k]:
                break
        index = + 1
        if index == p:
            value = 1
        else:
            value = 0
        auto_value.append(value)
        return inv + '_pdn' + str(p), auto_value


    def Trm(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)]
        auto_value = []
        for i in range(len(df)):
            trm_mean = list(df.loc[i, :])
        trm_mean.remove(np.nanmax(trm_mean))
        trm_mean.remove(np.nanmin(trm_mean))
        temp = np.nanmean(trm_mean)
        auto_value.append(temp)
        return inv + '_trm' + str(p), auto_value


    def Cmx(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)]
        auto_value = (df[inv + '1'] - np.nanmax(df, axis=1)) / (np.nanmax(df, axis=1) + 1e-10)
        return inv + '_cmx' + str(p), auto_value


    def Cmp(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)]
        auto_value = (df[inv + '1'] - np.nanmean(df, axis=1)) / (np.nanmean(df, axis=1) + 1e-10)
        return inv + '_cmp' + str(p), auto_value


    def Cnp(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)]
        auto_value = (df[inv + '1'] - np.nanmin(df, axis=1)) / (np.nanmin(df, axis=1) + 1e-10)
        return inv + '_cnp' + str(p), auto_value


    def Msx(self, inv, p):
        df = self.data.loc[:, inv + '1':inv + str(p)]
        df['_max'] = np.nanmax(df, axis=1)
        for i in range(1, p + 1):
            df[inv + str(i)] = list(df[inv + str(i)] == df['_max'])
        del df['_max']
        df_value = np.where(df == True, 1, 0)
        auto_value = []
        for i in range(len(df_value)):
            row_value = df_value[i, :]
        indexs = 1
        for j in row_value:
            if j == 1:
                break
        indexs += 1
        auto_value.append(indexs)
        return inv + '_msx' + str(p), auto_value


    def Rpp(self, inv, p):
        df1 = self.data.loc[:, inv + '1':inv + str(p)].values
        value1 = np.nanmean(df1, axis=1)
        df2 = self.data.loc[:, inv + str(p):inv + str(2 * p)].values
        value2 = np.nanmean(df2, axis=1)
        auto_value = value1 / (value2 + 1e-10)
        return inv + '_rpp' + str(p), auto_value


    def Dpp(self, inv, p):
        df1 = self.data.loc[:, inv + '1':inv + str(p)].values
        value1 = np.nanmean(df1, axis=1)
        df2 = self.data.loc[:, inv + str(p):inv + str(2 * p)].values
        value2 = np.nanmean(df2, axis=1)
        auto_value = value1 - value2
        return inv + '_dpp' + str(p), auto_value


    def Mpp(self, inv, p):
        df1 = self.data.loc[:, inv + '1':inv + str(p)].values
        value1 = np.nanmax(df1, axis=1)
        df2 = self.data.loc[:, inv + str(p):inv + str(2 * p)].values
        value2 = np.nanmax(df2, axis=1)
        auto_value = value1 / (value2 + 1e-10)
        return inv + '_mpp' + str(p), auto_value


    def Npp(self, inv, p):
        df1 = self.data.loc[:, inv + '1':inv + str(p)].values
        value1 = np.nanmin(df1, axis=1)
        df2 = self.data.loc[:, inv + str(p):inv + str(2 * p)].values
        value2 = np.nanmin(df2, axis=1)
        auto_value = value1 / (value2 + 1e-10)
        return inv + '_npp' + str(p), auto_value


data = pd.read_excel('textdata.xlsx')
FEATURE_LIST = ['ft','gt']
P_LIST = [1]
gen = feature_generation(data, FEATURE_LIST, P_LIST)
df = gen.fit_generate()



















