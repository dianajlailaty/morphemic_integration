import messaging
import morphemic
import time
import logging
import signal

# Libraries required for training and prediction
import os
import json
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import numpy as np
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
from time import sleep



METHOD = os.environ.get("METHOD", "prophet")
START_TOPIC = f"start_forecasting.{METHOD}"
STOP_TOPIC = f"stop_forecasting.{METHOD}"
PRED_TOPIC_PREF = f"intermediate_prediction.{METHOD}"
PREDICTION_CYCLE = 1  # minutes
APP_NAME = os.environ.get("APP_NAME", "demo")
AMQ_USER = os.environ.get("AMQ_USER", "admin")
AMQ_PASSWORD = os.environ.get("AMQ_PASSWORD", "admin")


class Prophet(morphemic.handler.ModelHandler,messaging.listener.MorphemicListener):
    id="prophet"
    metrics=set()

    def __init__(self):
        self._run =  False
        self.connector = messaging.morphemic.Connection('aaa','111', host='147.102.17.76', port=61610)
        #self.model = morphemic.model.Model(self)

    def run(self):
        logging.debug("setting up")
        self.connector.set_listener(self.id, self)
        self.connector.connect()
        self.connector.topic("start_forecasting.prophet", self.id)
        self.connector.topic("stop_forecasting.prophet", self.id)
        self.connector.topic("metrics_to_predict", self.id)
             
    def on_start_forecastig_prophet(self):
        logging.debug("Prophet started phorecasting")
        #read the model
        with open('Prophet.pkl', 'rb') as f:
            model = pickle.load(f)
            
            
        body=json.loads(frame.body)
        logging.debug(body)
        
        timestamp = body['timestamp']
        epoch_start = body["epoch_start"]
        prediction_horizon = body["prediction_horizon"]
        number_of_forward_predictions = body["number_of_forward_predictions"] 
        sent_metrics = body["metrics"]
        for metric in sent_metrics:
            if metric not in metrics:
                logging.debug("Subscribing to %s " % metric)
                #self.m.topic(metric,id)
                metrics.add(metric)
                
        logging.debug("THE METRICS")       
        logging.debug(metrics)
        #a=prediction_horizon 
        while(True):
            for metric in sent_metrics:
         
                predictions=predict(model , number_of_forward_predictions)
                logging.debug(predictions['yhat'].values)
            
                logging.debug("SENDING PREDICTION")
                #a=prediction_horizon 
                for v in predictions['yhat'].values.tolist():
                 
                    self.connector.send_to_topic('intermediate_prediction.prophet.'+metric,               
                    
                    {
                        "metricValue": v,
                        "level": 3,
                        "timestamp": int(time()),
                        "probability": 0.98,
                        "confidence_interval" : [12,20],
                        "horizon": 30,
                        "predictionTime" : int(epoch_start),
                        "refersTo": "MySQL_12345",
                        "cloud": "AWS-Dublin",
                        "provider": "AWS"
                        
                        })
                        
                    #a=a+prediction_horizon 
            #duration=time() - epoch_start
            epoch_start=epoch_start + prediction_horizon
            sleep(prediction_horizon)

    def on_train(self, frame):
        logging.debug("Here is the frame")
        logging.debug(frame)
        body=frame.body
        res = ast.literal_eval(body)
        #logging.debug(res[0])
        logging.debug("METRICS TO PREDICT") 
        logging.debug(res[0])
        
        for r in res:
            if r['metric'] not in self.metrics():
                logging.debug("Subscribing to %s " % r)    
                self.metrics.add(r['metric'])
        logging.debug(metrics)
        
        
        logging.debug("20 minutes")
        
        model=train("20 minutes")
        
        pkl_path = "Prophet.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        
        self.m.send_to_topic("training_models",
            {

            "metrics": list(metrics),

             "forecasting_method": "Prophet",

            "timestamp": int(time()/1000)

            }   
        )

    def on_start_forecastig_prophet(self, frame):
        logging.debug("Stop Forecasting %s" % frame)
        body=frame.body
        res = ast.literal_eval(body)
        #logging.debug(res[0])
        logging.debug("METRICS TO PREDICT") 
        logging.debug(res[0])
        
        for r in res:
            if r['metric'] in self.metrics:
                logging.debug("Un-subscribing from %s " % r)
                metrics.remove(r['metric'])
        logging.debug(metrics)
        logging.debug("STOP FORECASTING")   

    def start(self):
        logging.debug("Staring Prophet")
        self.run()
        self._run = True
        


def train(prediction_horizon):

    filename='demo'
    
    dataset= pd.read_csv(filename + ".csv")
    X= dataset[['name','time', 'countryCode', 'ipAddress', 'level',
     'producer']]
    y= dataset.iloc[:,6]
    #changing the names of the attributes
    prophet_dataset= pd.DataFrame()
    prophet_dataset['ds'] = pd.to_datetime(X["time"])
    prophet_dataset['y']=y
    prophet_dataset['ds'] = prophet_dataset['ds'].dt.tz_convert(None)
    
    #splitting
    test_percentage=0.2
    training_window_size=int(len(prophet_dataset)-(len(prophet_dataset)*test_percentage))
    train=prophet_dataset[:training_window_size]
    test=prophet_dataset[training_window_size:]
    
    #hyperparameter tuning and cross validation
    initial = '30 minutes'
    period = '15 minutes'
    horizon = prediction_horizon

    changepoint_prior_scale  = [0.1,0.2,0.3,0.4,0.5]
    n_changepoints = [15,20,25]
    #growth = ['logistic', 'linear'] 
    changepoint_range = [0.25, 0.5, 0.75]
    seasonality_mode = ["additive","multiplicative"]
    interval_width = [0.25, 0.5]  
    
    param_grid = {  #'changepoint_prior_scale' : changepoint_prior_scale,
                'n_changepoints' : n_changepoints,
                #'growth': growth, 
                'changepoint_range' : changepoint_range,
                #'seasonality_mode' : seasonality_mode,
                #'interval_width' : interval_width,   
                #'changepoints': changepoints, 
                #'yearly_seasonality': yearly_seasonality,
                #'weekly_seasonality': weekly_seasonality,
                #'daily_seasonality': daily_seasonality,
                #'holidays': holidays,
                #'seasonality_prior_scale': seasonality_prior_scale,
                #'holidays_prior_scale': holidays_prior_scale,    
              }
    grid = ParameterGrid(param_grid)
    cnt = 0
    for p in grid:
        cnt = cnt+1

    logging.debug(cnt)
    
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here
    maes = []  # Store the MAEs for each params here
    cutoffss = []
    df_cvs = []

    # Use cross validation to evaluate all parameters
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
    logging.debug(tuning_results)
    
    parameters = tuning_results.sort_values(by=['rmse'])
    parameters = parameters.reset_index(drop=True)
    logging.debug(parameters)
    
    #get the final model
    final_model = Prophet(
                     #seasonality_mode = parameters['seasonality_mode'][0],
                     #changepoint_prior_scale = parameters['changepoint_prior_scale'][0],
                     n_changepoints =  parameters['n_changepoints'][0],
                     #growth =  parameters['growth'][0],
                     changepoint_range = parameters['changepoint_range'][0],
                     #interval_width =  parameters['interval_width'][0],  
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
    return final_model    


def main():
    logging.getLogger().setLevel(logging.DEBUG)
    
    '''
    dataset_preprocessor = CSVData(APP_NAME)
    dataset_preprocessor.prepare_csv()
    logging.debug("dataset downloaded")
    '''
    m.set_listener(id,Listener(m,"/topic/metrics_to_predict.prophet"))
    m1.set_listener(id,StartForecastingListener(m1,"/topic/start_forecasting.prophet"))
    m2.set_listener(id,StopForecastingListener(m2,"/topic/stop_forecasting.prophet"))
    

    while True:
        pass