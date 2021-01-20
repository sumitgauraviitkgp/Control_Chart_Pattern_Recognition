import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
PI=math.pi

dataset=pd.read_csv('Raweyedata_drop.csv')



def normal_pattern(x_mean,x_std,y_mean,y_std):
    r=random.random()
    x1=x_mean+r*x_std
    y1=y_mean+r*y_std
    return x1,y1

def cyclic_pattern(x_mean,x_std,y_mean,y_std,t,T):
    PI=math.pi
    r=random.random()
    a_x=random.uniform(1.5*x_std,2.5*x_std)
    x2=x_mean+r*x_std+a_x*math.sin(2*PI*t/T)
    a_y=random.uniform(1.5*y_std,2.5*y_std)
    y2=y_mean+r*y_std+a_y*math.sin(2*PI*t/T)
    return x2,y2
def systematic_pattern(x_mean,x_std,y_mean,y_std,t):
    r=random.random()
    d_x=random.uniform(1*x_std,3*x_std)
    x3=x_mean+r*x_std+d_x*pow(-1,t)
    d_y=random.uniform(1*y_std,3*y_std)
    y3=y_mean+r*y_std+d_y*pow(-1,t)
    return x3,y3
def stratification_pattern(x_mean,x_std,y_mean,y_std):
    r=random.random()
    stdd_x=random.uniform(0.2*x_std,0.4*x_std)
    x4=x_mean+r*stdd_x
    stdd_y=random.uniform(0.2*y_std,0.4*y_std)
    y4=y_mean+r*stdd_y
    return x4,y4

def uptrend_pattern(x_mean,x_std,y_mean,y_std,t):
    r=random.random()
    g_x=random.uniform(0.05*x_std,0.25*x_std)
    x5=x_mean+r*x_std+t*g_x
    g_y=random.uniform(0.05*y_std,0.25*y_std)
    y5=y_mean+r*y_std+t*g_y
    return x5,y5

def downtrend_pattern(x_mean,x_std,y_mean,y_std,t):
    r=random.random()
    g_x=random.uniform(0.05*x_std,0.25*x_std)
    x6=x_mean+r*x_std-t*g_x
    g_y=random.uniform(0.05*y_std,0.25*y_std)
    y6=y_mean+r*y_std-t*g_y
    return x6,y6

def upshift_pattern(x_mean,x_std,y_mean,y_std,t):
    p=random.uniform(10,20)
    r=random.random()
    if t>=p:
        k=1
    else:
        k=0
    s_x=random.uniform(x_std,3*x_std)
    x7=x_mean+r*x_std+k*s_x
    s_y=random.uniform(y_std,3*y_std)
    x7=y_mean+r*y_std+k*s_y
    return x7,y7

def downshift_pattern(x_mean,x_std,y_mean,y_std,t):
    p=random.uniform(10,20)
    r=random.random()
    if t>=p:
        k=1
    else:
        k=0
    s_x=random.uniform(x_std,3*x_std)
    x8=x_mean+r*x_std-k*s_x
    s_y=random.uniform(y_std,3*y_std)
    x8=y_mean+r*y_std-k*s_y
    return x8,y8


x_point=np.zeros([87552,1])
y_point=np.zeros([87522,1])
series_id=np.zeros([87522,1])
group_id=np.zeros([2736,1])
label=np.zeros([2736,1])

#y_train=np.zeros([256,1])
for k in range(0,64):   
    df_x=dataset.iloc[k:k+32,2]
    df_y=dataset.iloc[k:k+32,3]

    x_mean=df_x.mean(axis=0)
    y_mean=df_y.mean(axis=0)

    x_std=df_x.std(axis=0)
    y_std=df_y.std(axis=0)
    for i in range(0,8):      
        for j in range(0,512):
            if i==0:   
                x_point[j],y_point[j]=normal_pattern(x_mean,x_std,y_mean,y_std)
                series_id[j]=i
            elif i==1:
                x_point[j+32],y_point[j+32]=cyclic_pattern(x_mean,x_std,y_mean,y_std,j+1,32)
                series_id[j+32]=i
            elif i==2:
                x_point[j+64],y_point[j+64]=systematic_pattern(x_mean,x_std,y_mean,y_std,j+1)
                series_id[j+64]=i
            elif i==3:
                x_point[j+96],y_point[j+96]=stratification_pattern(x_mean,x_std,y_mean,y_std)
                series_id[j+96]=i
            elif i==4:
                x_point[j+128],y_point[j+128]=uptrend_pattern(x_mean,x_std,y_mean,y_std,j+1)
                series_id[j+128]=i
            elif i==5:
                x_point[j+160],y_point[j+160]=downtrend_pattern(x_mean,x_std,y_mean,y_std,j+1)
                series_id[j+160]=i
            elif i==6:
                x_point[j+192],y_point[j+192]=upshift_pattern(x_mean,x_std,y_mean,y_std,j+1)
                series_id[j+192]=i
            elif i==7:
                x_point[j+224],y_point[j+224]=downshift_pattern(x_mean,x_std,y_mean,y_std,j+1)
                series_id[j+224]=i
            j=j+256
                    
    k=k+32

for n in range(0,16):
    for m in range(0,8):
        if m==0:
            label[n]='nor'
            group_id[n]=0
        elif m==1:
            label[n+1]='cyc'
            group_id[n]=1
        elif m==1:
            label[n+2]='sys'
            group_id[n]=2
        elif m==1:
            label[n+3]='str'
            group_id[n]=3
        elif m==1:
            label[n+4]='ut'
            group_id[n]=4
        elif m==1:
            label[n+5]='dt'
            group_id[n]=5
        elif m==1:
            label[n+6]='us'
            group_id[n]=6
        elif m==1:
            label[n+7]='ds'
            group_id[n]=7
    n=n+8

x=np.append(series_id,x_point,y_point,axis=1)
y=np.append(group_id,label)

