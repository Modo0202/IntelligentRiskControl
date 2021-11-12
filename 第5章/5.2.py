# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:16:02 2020

@author: meizihang
"""

#最近p个月，inv>0的月份数
def Num(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value=np.where(df>0,1,0).sum(axis=1)
    return inv+'_num'+str(p),auto_value

#最近p个月，inv=0的月份数
def Nmz(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value=np.where(df==0,1,0).sum(axis=1)
    return inv+'_nmz'+str(p),auto_value

#最近p个月，inv>0的月份数是否>=1     
def Evr(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    arr=np.where(df>0,1,0).sum(axis=1)
    auto_value = np.where(arr,1,0)
    return inv+'_evr'+str(p),auto_value    

#最近p个月，inv均值
def Avg(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value=np.nanmean(df,axis = 1 )
    return inv+'_avg'+str(p),auto_value    


#最近p个月，inv和
def Tot(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value=np.nansum(df,axis = 1)
    return inv+'_tot'+str(p),auto_value  


#最近(2,p+1)个月，inv和

def Tot2T(inv,p):
    df=data.loc[:,inv+'2':inv+str(p+1)]
    auto_value=df.sum(1)
    return inv+'_tot2t'+str(p),auto_value  


#最近p个月，inv最大值
def Max(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value=np.nanmax(df,axis = 1)
    return inv+'_max'+str(p),auto_value 


#最近p个月，inv最小值
def Min(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value=np.nanmin(df,axis = 1)
    return inv+'_min'+str(p),auto_value 

#最近p个月，最近一次inv>0到现在的月份数

def Msg(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    df_value=np.where(df>0,1,0)
    auto_value=[]
    for i in range(len(df_value)):
        row_value=df_value[i,:]
        if row_value.max()<=0:
            indexs='0'
            auto_value.append(indexs)
        else:
            indexs=1
            for j in row_value:
                if j>0:
                    break
                indexs+=1
            auto_value.append(indexs)
    return inv+'_msg'+str(p),auto_value
 

#最近p个月，最近一次inv=0到现在的月份数
def Msz(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    df_value=np.where(df==0,1,0)
    auto_value=[]
    for i in range(len(df_value)):
        row_value=df_value[i,:]
        if row_value.max()<=0:
            indexs='0'
            auto_value.append(indexs)
        else:
            indexs=1
            for j in row_value:
                if j>0:
                    break
                indexs+=1
            auto_value.append(indexs)
    return inv+'_msz'+str(p),auto_value   
    
#当月inv/(最近p个月inv的均值)
def Cav(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value = df[inv+'1']/np.nanmean(df,axis = 1 ) 
    return inv+'_cav'+str(p),auto_value 

#当月inv/(最近p个月inv的最小值)
def Cmn(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value = df[inv+'1']/np.nanmin(df,axis = 1 ) 
    return inv+'_cmn'+str(p),auto_value 

#最近p个月，每两个月间的inv的增长量的最大值
def Mai(inv,p):
    arr=np.array(data.loc[:,inv+'1':inv+str(p)])     
    auto_value = []
    for i in range(len(arr)):
        df_value = arr[i,:]
        value_lst = []
        for k in range(len(df_value)-1):
            minus = df_value[k] - df_value[k+1]
            value_lst.append(minus)
        auto_value.append(np.nanmax(value_lst))     
    return inv+'_mai'+str(p),auto_value 

#最近p个月，每两个月间的inv的减少量的最大值
def Mad(inv,p):
    arr=np.array(data.loc[:,inv+'1':inv+str(p)])      
    auto_value = []
    for i in range(len(arr)):
        df_value = arr[i,:]
        value_lst = []
        for k in range(len(df_value)-1):
            minus = df_value[k+1] - df_value[k]
            value_lst.append(minus)
        auto_value.append(np.nanmax(value_lst))     
    return inv+'_mad'+str(p),auto_value 

#最近p个月，inv的标准差
def Std(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value=np.nanvar(df,axis = 1)
    return inv+'_std'+str(p),auto_value 

    
#最近p个月，inv的变异系数
def Cva(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value=np.nanvar(df,axis = 1)/(np.nanmean(df,axis = 1 )+1e-10)
    return inv+'_cva'+str(p),auto_value 



#(当月inv) - (最近p个月inv的均值)
def Cmm(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value = df[inv+'1'] - np.nanmean(df,axis = 1 ) 
    return inv+'_cmm'+str(p),auto_value 

#(当月inv) - (最近p个月inv的最小值)
def Cnm(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value = df[inv+'1'] - np.nanmin(df,axis = 1 ) 
    return inv+'_cnm'+str(p),auto_value 


#(当月inv) - (最近p个月inv的最大值)
def Cxm(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value = df[inv+'1'] - np.nanmax(df,axis = 1 ) 
    return inv+'_cxm'+str(p),auto_value 


#（ (当月inv) - (最近p个月inv的最大值) ） / (最近p个月inv的最大值) ）
def Cxp(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    temp = np.nanmax(df,axis = 1 )
    auto_value = (df[inv+'1'] - temp )/ temp
    return inv+'_cxp'+str(p),auto_value 

#最近p个月，inv的极差
def Ran(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value = np.nanmax(df,axis = 1 )  -  np.nanmin(df,axis = 1 ) 
    return inv+'_ran'+str(p),auto_value 


#最近min( Time on book，p )个月中，后一个月相比于前一个月增长了的月份数
def Nci(inv,p):
    arr=np.array(data.loc[:,inv+'1':inv+str(p)])     
    auto_value = []
    for i in range(len(arr)):
        df_value = arr[i,:]
        value_lst = []
        for k in range(len(df_value)-1):
            minus = df_value[k] - df_value[k+1]
            value_lst.append(minus)           
        value_ng = np.where(np.array(value_lst)>0,1,0).sum()
        auto_value.append(np.nanmax(value_ng))     
    return inv+'_nci'+str(p),auto_value 
   

#最近min( Time on book，p )个月中，后一个月相比于前一个月减少了的月份数
def Ncd(inv,p):
    arr=np.array(data.loc[:,inv+'1':inv+str(p)])     
    auto_value = []
    for i in range(len(arr)):
        df_value = arr[i,:]
        value_lst = []
        for k in range(len(df_value)-1):
            minus = df_value[k] - df_value[k+1]
            value_lst.append(minus)           
        value_ng = np.where(np.array(value_lst)<0,1,0).sum()
        auto_value.append(np.nanmax(value_ng))     
    return inv+'_ncd'+str(p),auto_value 
           

#最近min( Time on book，p )个月中，相邻月份inv 相等的月份数
def Ncn(inv,p):
    arr=np.array(data.loc[:,inv+'1':inv+str(p)])     
    auto_value = []
    for i in range(len(arr)):
        df_value = arr[i,:]
        value_lst = []
        for k in range(len(df_value)-1):
            minus = df_value[k] - df_value[k+1]
            value_lst.append(minus)           
        value_ng = np.where(np.array(value_lst)==0,1,0).sum()
        auto_value.append(np.nanmax(value_ng))     
    return inv+'_ncn'+str(p),auto_value    
 
#If  最近min( Time on book，p )个月中，对任意月份i ，都有 inv[i] > inv[i+1] ，
#即严格递增，且inv > 0则flag = 1 Else flag = 0
def Bup(inv,p):
    arr=np.array(data.loc[:,inv+'1':inv+str(p)])     
    auto_value = []
    for i in range(len(arr)):
        df_value = arr[i,:]
        value_lst = []
        index = 0
        for k in range(len(df_value)-1):
            if df_value[k] > df_value[k+1]:
                break
            index =+ 1
        if index == p:            
            value= 1    
        else:
            value = 0
        auto_value.append(value)     
    return inv+'_bup'+str(p),auto_value   

#If  最近min( Time on book，p )个月中，对任意月份i ，都有 inv[i] < inv[i+1] ，
#即严格递减，且inv > 0则flag = 1 Else flag = 0
def Pdn(inv,p):
    arr=np.array(data.loc[:,inv+'1':inv+str(p)])     
    auto_value = []
    for i in range(len(arr)):
        df_value = arr[i,:]
        value_lst = []
        index = 0
        for k in range(len(df_value)-1):
            if df_value[k+1] > df_value[k]:
                break
            index =+ 1
        if index == p:            
            value= 1    
        else:
            value = 0
        auto_value.append(value)     
    return inv+'_pdn'+str(p),auto_value            



#最近min( Time on book，p )个月，inv的修建均值
def Trm(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value = []
    for i in range(len(df)):
        trm_mean = list(df.loc[i,:])
        trm_mean.remove(np.nanmax(trm_mean))
        trm_mean.remove(np.nanmin(trm_mean))
        temp=np.nanmean(trm_mean) 
        auto_value.append(temp)
    return inv+'_trm'+str(p),auto_value 

#当月inv / 最近p个月的inv中的最大值
def Cmx(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value = (df[inv+'1'] - np.nanmax(df,axis = 1 )) /np.nanmax(df,axis = 1 ) 
    return inv+'_cmx'+str(p),auto_value 

#( 当月inv - 最近p个月的inv均值 ) / inv均值
def Cmp(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value = (df[inv+'1'] - np.nanmean(df,axis = 1 )) /np.nanmean(df,axis = 1 ) 
    return inv+'_cmp'+str(p),auto_value 


#( 当月inv - 最近p个月的inv最小值 ) /inv最小值 
def Cnp(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    auto_value = (df[inv+'1'] - np.nanmin(df,axis = 1 )) /np.nanmin(df,axis = 1 ) 
    return inv+'_cnp'+str(p),auto_value 


#最近min( Time on book，p )个月取最大值的月份距现在的月份数
def Msx(inv,p):
    df=data.loc[:,inv+'1':inv+str(p)]
    df['_max'] = np.nanmax(df,axis = 1)
    for i in range(1,p+1):
        df[inv+str(i)] = list(df[inv+str(i)] == df['_max'])
    del df['_max']
    df_value = np.where(df==True,1,0)
    auto_value=[]
    for i in range(len(df_value)):
        row_value=df_value[i,:]
        indexs=1
        for j in row_value:
            if j == 1:
                break
            indexs+=1
        auto_value.append(indexs)
    return inv+'_msx'+str(p),auto_value


#最近p个月的均值/((p,2p)个月的inv均值)
def Rpp(inv,p):
    df1=data.loc[:,inv+'1':inv+str(p)]
    value1=np.nanmean(df1,axis = 1 )
    df2=data.loc[:,inv+str(p):inv+str(2*p)]
    value2=np.nanmean(df2,axis = 1 )   
    auto_value = value1/value2
    return inv+'_rpp'+str(p),auto_value    


#最近p个月的均值 - ((p,2p)个月的inv均值)
def Dpp(inv,p):
    df1=data.loc[:,inv+'1':inv+str(p)]
    value1=np.nanmean(df1,axis = 1 )
    df2=data.loc[:,inv+str(p):inv+str(2*p)]
    value2=np.nanmean(df2,axis = 1 )   
    auto_value = value1 - value2
    return inv+'_dpp'+str(p),auto_value   


#(最近p个月的inv最大值)/ (最近(p,2p)个月的inv最大值)
def Mpp(inv,p):
    df1=data.loc[:,inv+'1':inv+str(p)]
    value1=np.nanmax(df1,axis = 1 )
    df2=data.loc[:,inv+str(p):inv+str(2*p)]
    value2=np.nanmax(df2,axis = 1 )   
    auto_value = value1/value2
    return inv+'_mpp'+str(p),auto_value  

  
#(最近p个月的inv最小值)/ (最近(p,2p)个月的inv最小值)
def Npp(inv,p):
    df1=data.loc[:,inv+'1':inv+str(p)]
    value1=np.nanmin(df1,axis = 1 )
    df2=data.loc[:,inv+str(p):inv+str(2*p)]
    value2=np.nanmin(df2,axis = 1 )   
    auto_value = value1/value2
    return inv+'_npp'+str(p),auto_value  


#首先执行下面的全部函数 
       
#定义批量调用双参数的函数        
def auto_var2(inv,p):
    #global data_new
    try:
        columns_name,values=Num(inv,p)
        data_new[columns_name]=values
    except:
           print("Num PARSE ERROR",inv,p)
    try:
        columns_name,values=Nmz(inv,p)
        data_new[columns_name]=values
    except:
           print("Nmz PARSE ERROR",inv,p)
    try:
        columns_name,values=Evr(inv,p)
        data_new[columns_name]=values
    except:
           print("Evr PARSE ERROR",inv,p)
    try:
        columns_name,values=Avg(inv,p)
        data_new[columns_name]=values
    except:
           print("Avg PARSE ERROR",inv,p)
    try:
        columns_name,values=Tot(inv,p)
        data_new[columns_name]=values
    except:
        print("Tot PARSE ERROR",inv,p) 
    try:
        columns_name,values=Tot2T(inv,p)
        data_new[columns_name]=values
    except:
        print("Tot2T PARSE ERROR",inv,p)        
    try:
        columns_name,values=Max(inv,p)
        data_new[columns_name]=values
    except:
        print("Tot PARSE ERROR",inv,p)
    try:
        columns_name,values=Max(inv,p)
        data_new[columns_name]=values
    except:
        print("Max PARSE ERROR",inv,p)
    try:
        columns_name,values=Min(inv,p)
        data_new[columns_name]=values
    except:
        print("Min PARSE ERROR",inv,p)
    try:
        columns_name,values=Msg(inv,p)
        data_new[columns_name]=values
    except:
        print("Msg PARSE ERROR",inv,p)
    try:
        columns_name,values=Msz(inv,p)
        data_new[columns_name]=values
    except:
        print("Msz PARSE ERROR",inv,p)
    try:
        columns_name,values=Cav(inv,p)
        data_new[columns_name]=values
    except:
        print("Cav PARSE ERROR",inv,p)
    try:
        columns_name,values=Cmn(inv,p)
        data_new[columns_name]=values
    except:
        print("Cmn PARSE ERROR",inv,p)        
    try:
        columns_name,values=Std(inv,p)
        data_new[columns_name]=values
    except:
        print("Std PARSE ERROR",inv,p)   
    try:
        columns_name,values=Cva(inv,p)
        data_new[columns_name]=values
    except:
        print("Cva PARSE ERROR",inv,p)   
    try:
        columns_name,values=Cmm(inv,p)
        data_new[columns_name]=values
    except:
        print("Cmm PARSE ERROR",inv,p)  
    try:
        columns_name,values=Cnm(inv,p)
        data_new[columns_name]=values
    except:
        print("Cnm PARSE ERROR",inv,p)         
    try:
        columns_name,values=Cxm(inv,p)
        data_new[columns_name]=values
    except:
        print("Cxm PARSE ERROR",inv,p)          
    try:
        columns_name,values=Cxp(inv,p)
        data_new[columns_name]=values
    except:
        print("Cxp PARSE ERROR",inv,p)
    try:
        columns_name,values=Ran(inv,p)
        data_new[columns_name]=values
    except:
        print("Ran PARSE ERROR",inv,p)
    try:
        columns_name,values=Nci(inv,p)
        data_new[columns_name]=values
    except:
        print("Nci PARSE ERROR",inv,p)
    try:
        columns_name,values=Ncd(inv,p)
        data_new[columns_name]=values
    except:
        print("Ncd PARSE ERROR",inv,p)
    try:
        columns_name,values=Ncn(inv,p)
        data_new[columns_name]=values
    except:
        print("Ncn PARSE ERROR",inv,p)
    try:
        columns_name,values=Pdn(inv,p)
        data_new[columns_name]=values
    except:
        print("Pdn PARSE ERROR",inv,p) 
    try:
        columns_name,values=Cmx(inv,p)
        data_new[columns_name]=values
    except:
        print("Cmx PARSE ERROR",inv,p)         
    try:
        columns_name,values=Cmp(inv,p)
        data_new[columns_name]=values
    except:
        print("Cmp PARSE ERROR",inv,p)   
    try:
        columns_name,values=Cnp(inv,p)
        data_new[columns_name]=values
    except:
        print("Cnp PARSE ERROR",inv,p) 
    try:
        columns_name,values=Msx(inv,p)
        data_new[columns_name]=values
    except:
        print("Msx PARSE ERROR",inv,p)
    try:
        columns_name,values=Nci(inv,p)
        data_new[columns_name]=values
    except:
        print("Nci PARSE ERROR",inv,p)
    try:
        columns_name,values=Trm(inv,p)
        data_new[columns_name]=values
    except:
        print("Trm PARSE ERROR",inv,p)
    try:
        columns_name,values=Bup(inv,p)
        data_new[columns_name]=values
    except:
        print("Bup PARSE ERROR",inv,p)
    try:
        columns_name,values=Mai(inv,p)
        data_new[columns_name]=values
    except:
        print("Mai PARSE ERROR",inv,p)
    try:
        columns_name,values=Mad(inv,p)
        data_new[columns_name]=values
    except:
        print("Mad PARSE ERROR",inv,p)
    try:
        columns_name,values=Rpp(inv,p)
        data_new[columns_name]=values
    except:
        print("Rpp PARSE ERROR",inv,p)
    try:
        columns_name,values=Dpp(inv,p)
        data_new[columns_name]=values
    except:
        print("Dpp PARSE ERROR",inv,p)
    try:
        columns_name,values=Mpp(inv,p)
        data_new[columns_name]=values
    except:
        print("Mpp PARSE ERROR",inv,p)
    try:
        columns_name,values=Npp(inv,p)
        data_new[columns_name]=values
    except:
        print("Npp PARSE ERROR",inv,p)
    return data_new.columns.size


data = pd.read_excel('textdata.xlsx')  
data.head()

auto_data = pd.DataFrame()  
for p in range(1, 12):  
    for inv in ['ft', 'gt']:  
        auto_var2(inv, p)  

data_new.shape


x['n1'] = x.apply(lambda x:1 if x.amount_tot>9614.5 \                          
                      and coupon_amount_cnt<=6.0 else 0)  
x['n2'] = x.apply(lambda x:1 if x.amount_tot>9614.5 \  
                      and coupon_amount_cnt>6.0 else 0)
























