# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


###############################################################################
# Functions
###############################################################################
# Forward and backward mean
def  ward_mean(data, parameter):
    ward_data=[]
    den = np.ones(len(data)-1)
    for i in range(1, len(data)):
        mean=np.sum(data[:i]/len(data[:i]))
        np.append(ward_data, mean)                    
    return ward_data

# Moving averages:
def moving_avg(x,n):
    return np.convolve(x, np.ones(n), 'valid')/n

# Forward and backward mean
def  ward_mean(data, parameter):
    '''

    Parameters
    ----------
    data : np.array
        1D Data array (Ndim) where perform the forward or backward average
    parameter : string
        'f': forward average
        'b': backward average
    Returns
    -------
    averaged_data : np.array
        1D veraged Data array (N-1 dim)

    '''
    averaged_data=[]
    if parameter == 'f':
        print('Performing forward average...')
    elif parameter == 'b':
        print('Performing backward average...')
    else:
        print('Parameter must be a string:  f or b for Forward or Backward mean')
    # Loop all over data array
    for i in range(2, len(data)+1):
        if parameter == 'f':
            mean=np.sum(data[:i])/i 
        elif parameter == 'b':
            mean=np.sum(data[len(data)-i:])/i
        else:
            print('parameter error!')
            break
        # Append data to outcomes array
        averaged_data = np.append(averaged_data, mean)
    return averaged_data

# Remove outliers:
def removeOutliers(x, outliers_cost):
    a=np.array(x)
    upper_quartile=np.percentile(a,75)
    lower_quartile=np.percentile(a,25)
    IQR=(upper_quartile-lower_quartile)*outliers_cost
    quartileSet=(lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
        else:
            resultList.append(np.nan)
    return resultList

# np.nan interpolation
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


# PARAMETERS
rho_w=1000 # kg/m^3
rho_s=2650 # kg/m^3

# Initialize Arrays
qs = [] # Solid discharge array

w_dir=os.getcwd()
file_name = 'output_Preparazione_solida_q10rgm1.txt'
file = os.path.join(w_dir, 'raw_solid', file_name)
path_out = os.path.join(w_dir, 'output_files')
run = 'q10rgm1'
# Set directory
if os.path.exists(path_out):
    pass
else:
    os.mkdir(path_out)

# Read input file:
df = pd.read_csv(file
                 ,sep="\t", skiprows=4
                  ,names=['time', 'Weight_kg', 'dis_lpers']
                 ,engine="python"
                 # ,index_col=(0)
                 # ,dtype={'Istante [H:M]':np.detetime64, 'Peso [kg]':np.float64, 'Portata [l/s]':np.float64}
                 )

# Operation on dataframe and np.array conversion:
df['time'] = pd.to_datetime(df['time']) # Convert a sinmgle column from dtype=object to dtype=detetime64

df['delta_weight_kg'] = df.Weight_kg.diff().fillna(0) # Perform difference between weight for each timestep
df['delta'] = (df['time']-df['time'].shift()).fillna(pd.Timedelta('0 days')) # Perform the difference between timesteps
df['delta_sec'] = df['delta'].dt.total_seconds() # Create a row with seconds between timesteps as integer
columns=['delta_sec','Weight_kg', 'dis_lpers'] # Select the columns to export
df1=df[columns] # create a new dataframe with extraction colums
df_array=df1.to_numpy() # convert Pandas dataframe to np.array

# NB: If the run go through the 00:00:00 the dataframe show a ~-86320 seconds in the corrisponden cell. Delate it!
# data array structure:
# delta_time [s]    weight [kg]    dis [l/s]
index = np.where(abs(df_array[:,0])>86000) # find index of time value where timestep go through 00:00:00
for i in range(0, len(index)):
    i=int(index[i])
    df_array[i,0] = np.mean(df_array[i-10:i-1,0]) # Correction: mean of the last 10 values of the time serie

# Data resampling and interpolation:
# Build time progression array in seconds
time=[] # initialize Variable
for i in range(len(df_array[:,0])):
    time=np.append(time, np.sum(df_array[:i,0]))
# Data interpolation
t = np.arange(0, 10*len(df_array[:,0]), 10) # Target interpolation array
weight=np.interp(t,time, df_array[:,1]) # Interpolation weight [kg]
disc=np.interp(t,time,df_array[:,2]) # Interpolate discharge [l/s]

# Calculate trendline for weight data excluding 5% extreems of dataset
# linear regression y=mx+q, 
m, q = np.polyfit(t[int(len(t)/20):-int(len(t)/20)], weight[int(len(t)/20):-int(len(t)/20)], 1)

# Weight detrending:
w_detrend = weight-(m*t+q)
np.savetxt(os.path.join(path_out, 'weight_detrend.txt'), w_detrend, fmt='%.3f', delimiter=',')

# Perform the difference in weight over timesteps
delta_w=[0]
delta_w=np.append(delta_w, weight[1:]-weight[:-1])
t_steps = np.arange(10,(len(t)+1)*10, 10)

# Perform dry weight calculation
d_w = np.array(delta_w/(1-rho_w/rho_s)) # Real dry sand weight
np.savetxt(os.path.join(path_out, 'd_w.txt'), d_w, fmt='%.3f', delimiter=',')

# Perform qs calculation as d_w/d_t
q_s=np.divide(d_w, t_steps)*1000 # Calculate qs [g/s]
q_s=np.where(np.isnan(q_s),0, q_s)
# Create dataset as: time[s], weight[kg], delta_weight[kg], q_s [g/s] discharge[l/s]
data = np.stack((t, weight, d_w, q_s, disc), axis=1)

# Spike removal
# Remove outliers:
d_w = np.array(removeOutliers(d_w, 1.5))
nan_count = np.sum(np.isnan(d_w))
print()
print(nan_count, ' outlier points - equal to ', format(nan_count/len(t_steps)*100, ".2f"), '% of the entire dataset.')
print()
np.savetxt(os.path.join(path_out, 'd_w_outliers_removal.txt'), d_w, fmt='%.3f', delimiter=',')

# Nan interpolation:
nans, x= nan_helper(d_w)
d_w[nans] = np.interp(x(nans), x(~nans), d_w[~nans])

np.savetxt(os.path.join(path_out, 'd_w_outliers_removal.txt'), d_w, fmt='%.3f', delimiter=',')

# Perform moving average calculation
dw_mov_vgd=moving_avg(d_w, 6)


# Calculate qs [g/s] (dt 10s by default) kg/10s --> 1000g/10s --> 100g/s
qs = dw_mov_vgd*100

# Calculate forward and backward average
qs_fwd = ward_mean(qs, 'f')
qs_bwd = ward_mean(qs, 'b')





###############################################################################
# PLOTS
###############################################################################

# Plot of weight over time and linear trendline
fig1, ax1 = plt.subplots()
ax1.plot(t, weight)
ax1.plot(t[int(len(t)/10):-int(len(t)/10)], m*t[int(len(t)/10):-int(len(t)/10)]+q)
ax1.set_title('Weight '+ run)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Weight [Kg]')
plt.show()

  
fig2, axes = plt.subplots(2,1)
axes[0].plot(t[0:len(d_w)], d_w)
axes[0].set_title('raw signal '+ run)
axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('delta_Weight [Kg]')

sns.set_theme(style="whitegrid")
sns.boxplot(data=d_w)



# TODO
#Plots with sublots with weight and delta weight 

np.savetxt(os.path.join(path_out, 'weight.txt'), weight, fmt='%.3f', delimiter=',')
np.savetxt(os.path.join(path_out, 'd_w_moving_avg.txt'), d_w, fmt='%.3f', delimiter=',')

























