# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 09:01:14 2025

@author: lemon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def weighted_mean(x,sigma):  #equation to determine the weighted mean and weighted standard deviation for a set of data points, with weighting of 1/std**2
    weights=1/sigma**2 
    mean=np.sum(x*weights)/np.sum(weights)
    mean_unc=np.sqrt(1/np.sum(weights))
    return mean, mean_unc

R=6730
R_std=1
def C_std(C, V_g_std , V_g , V_x_std , V_x, theta , theta_std,R_std,R):
    V_g_ratio=(V_g_std/V_g)
    V_x_ratio=V_x_std/V_x
    theta_err=theta_std/np.tan(theta)
    R_ratio=(R_std/R)
    return C*np.sqrt((V_g_ratio)**2 + (V_x_ratio)**2 +(theta_err)**2 + (R_ratio)**2)

C_mean_arr=[]
C_std_arr=[]
f_arr=[]
#1-3 is for 25kHz, 4-6 is for 50kHz, 7-9 is for 75kHz, 10-12 is for 100kHz
# mean is row index 8, std is row index 9
for i in range(1,13):
    cap_data=pd.read_csv(f"C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR3_{i}.CSV",skiprows=12,nrows=10,usecols=[2,4,6])
    if i<4:
        f=25000
        f_arr.append(f)
    elif i>3 and i<7:
        f=50000
        f_arr.append(f)
    elif i>6 and i<10:
        f=75000
        f_arr.append(f)
    else:
        f=100000
        f_arr.append(f)
        
    V_g=cap_data.iloc[8,0]
    V_g_std=cap_data.iloc[9,0]
    V_x=cap_data.iloc[8,1]
    V_x_std=cap_data.iloc[9,1]
    theta=(cap_data.iloc[8,2])*(np.pi/180)
    theta_std=(cap_data.iloc[9,2]*np.pi/180)
    
    
    C_approx=(V_g*np.sin(theta)) / (2*np.pi*f*R*V_x) #Approximation made using binomial expansion
    C_err=C_std(C_approx,V_g_std,V_g,V_x_std,V_x,theta,theta_std,R_std,R)
    C_mean_arr.append(C_approx)
    C_std_arr.append(C_err)

    
#THE BELOW LINES MAY BE COMMENTED OUT TO CALCULATE A FULLY ACCURATE VALUE OF C, BUT THE ERROR IS INCREDIBLY SMALL, SO BINOMIAL APPROXIMATION IS USED INSTEAD    
    #D=((V_g*np.cos(theta))-V_x)**2 +(V_g*np.sin(theta))**2
    #C_acc=np.sqrt(D)/(2*np.pi*f*R*V_x)
    #C_percent=np.abs((C_acc-C_approx/C_acc))*100      
    #C_err=100-C_percent           
    #print('The error in C between the binomial approximation and the full equation for C in trial', f' {i} is {C_err:.4f} percent ')
    
C_mean_arr = np.array(C_mean_arr, dtype=float)
C_mean_arr=(C_mean_arr) * 10**12 #pF
f_arr=np.array(f_arr,dtype=float)
f_arr=f_arr/1000 #kHz
C_std_arr = np.array(C_std_arr, dtype=float)
C_std_arr=C_std_arr*10**12

plt.errorbar(f_arr,C_mean_arr,label='Data points',barsabove=True,fmt='x',yerr=C_std_arr,capsize=5,elinewidth=2,markeredgewidth=0.5)
fit_params=np.polyfit(f_arr,C_mean_arr,deg=1)
data_fit=np.polyval(fit_params,f_arr)
plt.plot(f_arr,data_fit,label='Fitted line')
plt.title('Effective Capacitance against Frequency')
plt.legend()
plt.xlabel('Frequency (kHz)')
plt.ylabel('Capacitance (pF) ')
C_at_2=np.polyval(fit_params,2)
print(C_at_2) #pF

w_mean,w_std=weighted_mean(C_mean_arr,C_std_arr)

print(w_mean,w_std) #pF
    

    
    
    
    
