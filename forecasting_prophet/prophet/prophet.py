import pandas as pd
import numpy as np
import logging
import itertools
import matplotlib.pyplot as plt
import plotly.offline as pyoff
import plotly.graph_objs as go
from sklearn import preprocessing
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from math import log
from math import exp
pd.set_option('display.max_row', 500)
import itertools
from sklearn.model_selection import ParameterGrid
#from dataset_maker import CSVData
from time import time
from datetime import datetime
import ast
import pickle
import json
import os
import math

def train(metric):
    #loading the dataset
    filename=os.environ.get("APP_NAME")
    dataset= pd.read_csv(filename + ".csv")
  
    #changing the names and the format of the attributes
    prophet_dataset= pd.DataFrame()
    prophet_dataset['ds'] = dataset["time"]
    prophet_dataset['y']=dataset[metric]
    prophet_dataset['y'] = pd.to_numeric(prophet_dataset['y'], errors='coerce')
    for  i in range (0,len(prophet_dataset['ds'])):
        prophet_dataset['ds'] [i]= datetime.fromtimestamp(prophet_dataset['ds'] [i])

    for i in range(0,len(prophet_dataset['y'])):
        if math.isnan(prophet_dataset['y'][i]):
           # print("true")
            prophet_dataset['y'][i] = prophet_dataset['y'].mean()

    size = len(prophet_dataset)
    
    logging.debug("STARTED TRAINING FOR: "+ metric)

    #splitting to train and test
    #test_percentage=0.2
    #training_window_size=int(len(prophet_dataset)-(len(prophet_dataset)*test_percentage))
    train=prophet_dataset[:size]
    #test=prophet_dataset[training_window_size:]
    
    #hyperparameter tuning and cross validation
    #should be generic
    t1 = dataset['time'][0]
    t2 = dataset['time'][size-1]
    timeDiffInSec = int(t2-t1)
    timeDiffInMin = timeDiffInSec/60
    
    init = int(timeDiffInMin/3)
    h = int(init/3)
    initial = str(init) + " minutes"
    horizon = str(h) + " minutes"
    period = str(h/2) + " minutes"
    
    logging.debug(initial)
    logging.debug(horizon)
    logging.debug(period)
    
    #initial = '10 minutes'
    #period = '5 minutes'
    #horizon = prediction_horizon
    '''
    changepoint_prior_scale  = [0.1,0.2,0.3,0.4,0.5]
    n_changepoints = [15,20,25]
    #growth = ['logistic', 'linear'] 
    changepoint_range = [0.25, 0.5, 0.75]
    seasonality_mode = ["additive","multiplicative"]
    interval_width = [0.75,0.8,0.85]  
    '''
    changepoint_prior_scale  = [0.1]
    n_changepoints = [25]
    #growth = ['logistic', 'linear'] 
    changepoint_range = [0.75]
    seasonality_mode = ["additive","multiplicative"]
    interval_width = [0.8]
    
    param_grid = {  #'changepoint_prior_scale' : changepoint_prior_scale,
                'n_changepoints' : n_changepoints,
                #'growth': growth, 
                'changepoint_range' : changepoint_range,
                #'seasonality_mode' : seasonality_mode,
                'interval_width' : interval_width,   
              }
    grid = ParameterGrid(param_grid)
    cnt = 0
    for p in grid:
        cnt = cnt+1

    
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here
    maes = []  # Store the MAEs for each params here
    cutoffss = []
    df_cvs = []

    #use cross validation to evaluate all parameters
    for params in all_params:
        m = Prophet(**params).fit(train)  # Fit model with given params
        df_cv = cross_validation(m, initial=initial, period = period, horizon = horizon)
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])
        maes.append(df_p['mae'].values[0])
        cutoffs = df_cv.groupby('cutoff').mean().reset_index()['cutoff']
        cutoff = df_cv['cutoff'].unique()[0]
        df_cv = df_cv[df_cv['cutoff'].values == cutoff]
        cutoffss.append(cutoffs)
        df_cvs.append(df_cv)
        
    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    tuning_results['mae'] = maes
    tuning_results['cutoffs'] = cutoffss
    tuning_results['df_cv'] = df_cvs

    #global prob
    parameters = tuning_results.sort_values(by=['rmse'])
    parameters = parameters.reset_index(drop=True)
    prob = parameters['interval_width'][0]

    
    #get the final model
    final_model = Prophet(
                     #seasonality_mode = parameters['seasonality_mode'][0],
                     #changepoint_prior_scale = parameters['changepoint_prior_scale'][0],
                     n_changepoints =  parameters['n_changepoints'][0],
                     #growth =  parameters['growth'][0],
                     changepoint_range = parameters['changepoint_range'][0],
                     interval_width =  parameters['interval_width'][0],  
                     #changepoints = parameters['changepoints'][0],
                     #yearly_seasonality = parameters['yearly_seasonality'][0],
                     #weekly_seasonality =  parameters['weekly_seasonality'][0],
                     #daily_seasonality = parameters['daily_seasonality'][0],
                     #holidays: parameters['holidays'][0],
                     #seasonality_prior_scale = parameters['seasonality_prior_scale'][0],
                     #holidays_prior_scale = parameters['holidays_prior_scale'][0],
                     #mcmc_samples = parameters['mcmc_samples'][0]
                      )
    final_model.fit(train)
    #probabilities[metric] = prob
    #checking if probabilities file exist
    if(os.path.isfile('prob_file.npy')):
        probs = np.load('prob_file.npy',allow_pickle='TRUE').item()
        probs[metric] = prob

    else:
        probs=dict()
        probs[metric] = prob

    #writing probabilities in a file
    np.save('prob_file.npy', probs) 
    return final_model
        
def predict(model , number_of_forward_predictions , prediction_horizon , epoch_start):
    future = list()
    for i in range(1, number_of_forward_predictions+1):
        dateInSec = epoch_start + i*prediction_horizon
        date=datetime.fromtimestamp(dateInSec)
        future.append(date)
    future = pd.DataFrame(future)
    future.columns = ['ds']
    forecast = model.predict(future)
    return forecast
