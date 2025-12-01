# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 15:07:45 2025

@author: lemon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def exp_decay_t(t,V0,tau):
    return V0* np.exp(-t/tau)

def weighted_mean(x,sigma):  #equation to determine the weighted mean and weighted standard deviation for a set of data points, with weighting of 1/std**2
    weights=1/sigma**2 
    mean=np.sum(x*weights)/np.sum(weights)
    mean_unc=np.sqrt(1/np.sum(weights))
    return mean, mean_unc
R_res=98990  #Resistance of the resistor
R_scope=10**6  #Resistor of the oscilliscope 
R=(1/R_res + 1/R_scope)**-1 #Total resistance 
R_unc=10
C_array=[] # Defining an empty list to store values of capacitance and their respective uncertainties
C_array_unc=[]

cap_data_1=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_1.CSV")
cap_data_2=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_2.CSV")
cap_data_3=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_3.CSV")
cap_data_4=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_4.CSV")
cap_data_5=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_5.CSV")

#The commented read.csv files are for convenience when working from my laptop and then working from school computers

# cap_data_1=pd.read_csv("C:/Users/ns1925/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_1.CSV")
# cap_data_2=pd.read_csv("C:/Users/ns1925/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_2.CSV")
# cap_data_3=pd.read_csv("C:/Users/ns1925/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_3.CSV")
# cap_data_4=pd.read_csv("C:/Users/ns1925/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_4.CSV")
# cap_data_5=pd.read_csv("C:/Users/ns1925/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_5.CSV")

cap_data_files=[cap_data_2,cap_data_3,cap_data_4,cap_data_5]

x_t=cap_data_1.iloc[:,0].to_numpy()  #Convert x data and y data to 1D numpy array
y_V=cap_data_1.iloc[:,1].to_numpy()

t = x_t - x_t[0]    # now t[0] = 0
#Array of initial fit parameters
V_guess=y_V.max()
target=V_guess/np.e
idx=np.argmin(np.abs(y_V-target))
tau=t[idx]

p0=[V_guess,tau]

fit_params,fit_cov=curve_fit(exp_decay_t, t, y_V, p0=p0,maxfev=1000000,)
data_fit=exp_decay_t(t,*fit_params)
plt.plot(x_t,y_V,label='Raw data')
plt.title('98990 \u03A9 trial 1')
plt.plot(x_t,data_fit,label='Fitted curve')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show()








tau_fit=fit_params[1] #Yields the value of time constant and its assosciated uncertainty from fit parameter array, and the covariance matrix
tau_unc=np.sqrt(fit_cov[1,1]) 
C=tau_fit/R
C_array.append(C) 
C_unc=np.sqrt(((tau_unc)/R)**2 + ((tau_fit*R_unc)/R**2)**2) #Uncertainty in C propogated from uncertainty in R and tau
C_array_unc.append(C_unc)

for i in range (0,4):  #Reading through each cap_data file in the list 
    cap_data=cap_data_files[i]
    x_t=cap_data.iloc[:,0].to_numpy()  #Convert x data and y data to 1D numpy array
    y_V=cap_data.iloc[:,1].to_numpy()
    t = x_t - x_t[0]    # now t[0] = 0
    #Array of initial fit parameters
    V_guess=y_V.max()
    target=V_guess/np.e
    idx=np.argmin(np.abs(y_V-target))
    tau=t[idx]
    
    p0=[V_guess,tau]
    
    fit_params,fit_cov=curve_fit(exp_decay_t, t, y_V, p0=p0,maxfev=1000000,)
    data_fit=exp_decay_t(t,*fit_params)
    plt.plot(x_t,y_V,label='Raw data')
    plt.title(f'98990 \u03A9 trial {i+1}')
    plt.plot(x_t,data_fit,label='Fitted curve')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.show()
    
    tau_fit=fit_params[1] #Yields the value of time constant and its assosciated uncertainty from fit parameter array, and the covariance matrix
    tau_unc=np.sqrt(fit_cov[1,1]) 
    C=tau_fit/R
    C_array.append(C) 
    C_unc=np.sqrt(((tau_unc)/R)**2 + ((tau_fit*R_unc)/R**2)**2) #Uncertainty in C propogated from uncertainty in R and tau
    C_array_unc.append(C_unc)


#%%

R_res=9895  #Resistance of the resistor
R_scope=10**6  #Resistor of the oscilliscope 
R=(1/R_res + 1/R_scope)**-1 #Total resistance 
R_unc=1

cap_data_6=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_6.CSV")
cap_data_7=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_7.CSV")
cap_data_8=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_8.CSV")
cap_data_9=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_9.CSV")
cap_data_10=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_10.CSV",skiprows=4300)

cap_data_files=[cap_data_6,cap_data_7,cap_data_8,cap_data_9,cap_data_10]
for i in range (0,5):  #Reading through each cap_data file in the list 
    cap_data=cap_data_files[i]
    x_t=cap_data.iloc[:,0].to_numpy()  #Convert x data and y data to 1D numpy array
    y_V=cap_data.iloc[:,1].to_numpy()
    t = x_t - x_t[0]    # now t[0] = 0
    #Array of initial fit parameters
    V_guess=y_V.max()
    target=V_guess/np.e
    idx=np.argmin(np.abs(y_V-target))
    tau=t[idx]
    
    p0=[V_guess,tau]
    
    fit_params,fit_cov=curve_fit(exp_decay_t, t, y_V, p0=p0,maxfev=1000000,)
    data_fit=exp_decay_t(t,*fit_params)
    plt.plot(x_t,y_V,label='Raw data')
    plt.title(f'9895 \u03A9 trial {i+1}')
    plt.plot(x_t,data_fit,label='Fitted curve')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.show()
   
 
    
    
    tau_fit=fit_params[1] #Yields the value of time constant and its assosciated uncertainty from fit parameter array, and the covariance matrix
    tau_unc=np.sqrt(fit_cov[1,1]) 
    C=tau_fit/R
    C_array.append(C) 
    C_unc=np.sqrt(((tau_unc)/R)**2 + ((tau_fit*R_unc)/R**2)**2) #Uncertainty in C propogated from uncertainty in R and tau
    C_array_unc.append(C_unc)
    
    
    
#%%

R_res=6730  #Resistance of the resistor
R_scope=10**6  #Resistor of the oscilliscope 
R=(1/R_res + 1/R_scope)**-1 #Total resistance 
R_unc=1

cap_data_11=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_11.CSV",skiprows=15000,nrows=40000)
cap_data_12=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_12.CSV",skiprows=4000,nrows=40000)
cap_data_13=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_13.CSV",skiprows=500)
cap_data_14=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_14.CSV")
cap_data_15=pd.read_csv("C:/Users/lemon/OneDrive - Imperial College London/Lab python code/Data for capacitance experiment 2/CIR2_15.CSV")


cap_data_files=[cap_data_11,cap_data_12,cap_data_13,cap_data_14,cap_data_15]
for i in range (0,5):  #Reading through each cap_data file in the list 
    cap_data=cap_data_files[i]
    x_t=cap_data.iloc[:,0].to_numpy()  #Convert x data and y data to 1D numpy array
    y_V=cap_data.iloc[:,1].to_numpy()
    t = x_t - x_t[0]    # now t[0] = 0
    #Array of initial fit parameters
    V_guess=y_V.max()
    target=V_guess/np.e
    idx=np.argmin(np.abs(y_V-target))
    tau=t[idx]
    
    p0=[V_guess,tau]
    
    fit_params,fit_cov=curve_fit(exp_decay_t, t, y_V, p0=p0,maxfev=1000000,)
    data_fit=exp_decay_t(t,*fit_params)
    plt.plot(x_t,y_V,label='Raw data')
    plt.title(f'6730 \u03A9 trial {i+1}')
    plt.plot(x_t,data_fit,label='Fitted curve')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.show()
    
    
    tau_fit=fit_params[1] #Yields the value of time constant and its assosciated uncertainty from fit parameter array, and the covariance matrix
    tau_unc=np.sqrt(fit_cov[1,1]) 
    C=tau_fit/R
    C_array.append(C) 
    C_unc=np.sqrt(((tau_unc)/R)**2 + ((tau_fit*R_unc)/R**2)**2) #Uncertainty in C propogated from uncertainty in R and tau
    C_array_unc.append(C_unc)
    
print(C_array)
C_array = np.array(C_array, dtype=float)
C_array_unc = np.array(C_array_unc, dtype=float)
w_mean,w_unc=weighted_mean(C_array,C_array_unc)
print(f'The weighted mean of capacitance across the 5 trials is ({w_mean*(10**12)}+-{w_unc*(10**12)})picofarads')
