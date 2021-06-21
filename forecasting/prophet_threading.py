import messaging
import os
import morphemic
import time
import logging
import signal
import json

import pandas as pd
import numpy as np
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
from dataset_maker import CSVData
from time import time
from datetime import datetime
import ast
import pickle
from time import sleep
import os.path
import threading

metrics = set()
#metrics = {"cpu_usage"}
id = "prophet"
probabilities = dict()
flags=dict()
flags = {'cpu_usage': 0 , 'latency': 0 , 'memory': 0 ,  'response_time': 0}
predictionTimes=dict()
models=dict()
#epoch_start = dict()
#probabilities["cpu_usage"] = 0.85

METHOD = os.environ.get("METHOD", "prophet")
START_TOPIC = f"start_forecasting.{METHOD}"
STOP_TOPIC = f"stop_forecasting.{METHOD}"
PRED_TOPIC_PREF = f"intermediate_prediction.{METHOD}"
PREDICTION_CYCLE = 1  # minutes
APP_NAME = os.environ.get("APP_NAME", "demo")
AMQ_USER = os.environ.get("AMQ_USER", "admin")
AMQ_PASSWORD = os.environ.get("AMQ_PASSWORD", "admin")

def worker(self,frame,metric):
    
    logging.debug("START FORECASTING")

    logging.debug("I AM HERE AGAIN TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
        
    body=json.loads(frame.body)

    timestamp = body['timestamp']

    prediction_horizon = body["prediction_horizon"]
    number_of_forward_predictions = body["number_of_forward_predictions"] 
   
        
    logging.debug(metric)
    epoch_start= body["epoch_start"]
    predictionTimes[metric] = epoch_start
    if metric not in metrics:
        if  os.path.isfile('prophet_'+metric+".pkl"):  
            logging.debug("Subscribing to %s " % metric)
            #self.m.topic(metric,id)
            metrics.add(metric)
                
            
    logging.debug("THE METRICS")       
    logging.debug(metrics)
    #a=prediction_horizon 


    while(True):
        
        if flags[metric] == 0:
            #logging.debug("THIS IS THE FIRST EPOCH START for "+ metric)
            #logging.debug(epoch_start)
            epoch_start = predictionTimes[metric]
            flags[metric] = 1
            #load the model
        with open("prophet_"+metric+".pkl", 'rb') as f:
            models[metric] = pickle.load(f)
    
        
        predictions=predict(models[metric] , number_of_forward_predictions , prediction_horizon , epoch_start)
        yhats = predictions['yhat'].values.tolist()
        yhat_lowers = predictions['yhat_lower'].values.tolist()
        yhat_uppers = predictions['yhat_upper'].values.tolist()
        
        prediction_time= epoch_start+ prediction_horizon
        logging.debug("THIS IS THE FIRST PREDICTION  TIME for "+ metric)
        logging.debug(prediction_time)
        timestamp = int(time())
        
        #read probabilities file
        probs = np.load('prob_file.npy' , allow_pickle='TRUE').item()
        logging.debug("probs")

    
        logging.debug("SENDING PREDICTION")

        
        for k in range(0,len(predictions['yhat'].values.tolist())):
            yhat = yhats[k]
            yhat_lower = yhat_lowers[k]
            yhat_upper = yhat_uppers[k]
            
            self.m1.send_to_topic('intermediate_prediction.prophet.'+metric,               
            
            {
                "metricValue": yhat,
                "level": 3,
                "timestamp": timestamp,
                "probability": probs[metric],
                "confidence_interval" : [yhat_lower,yhat_upper],
                "horizon": prediction_horizon,
                "predictionTime" : int(prediction_time),
                "refersTo": "todo",
                "cloud": "todo",
                "provider": "todo"  
                })
                
            prediction_time=prediction_time + prediction_horizon
        
        epoch_start = epoch_start+ prediction_horizon
        sleep(prediction_horizon)
        


def train(metric):

    #loading the dataset
    filename='demo'
    dataset= pd.read_csv(filename + ".csv")
  
    #changing the names and the format of the attributes
    prophet_dataset= pd.DataFrame()
    prophet_dataset['ds'] = dataset["time"]
    logging.debug(prophet_dataset['ds'] )
    prophet_dataset['y']=dataset[metric]
    for  i in range (0,len(prophet_dataset['ds'])):
        prophet_dataset['ds'] [i]= datetime.fromtimestamp(prophet_dataset['ds'] [i])
    
    #logging.debug("THE TRAINING TIMESTAMPS")
    #logging.debug(prophet_dataset['ds'])
    size = len(prophet_dataset)
    
    logging.debug("STARTED TRAINING FOR: "+ metric)

    #splitting to train and test
    #test_percentage=0.2
    #training_window_size=int(len(prophet_dataset)-(len(prophet_dataset)*test_percentage))
    train=prophet_dataset[:size]
    #logging.debug("THE TRAINING TIMESTAMPS")
    #logging.debug(train['ds'])
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
    logging.debug(cnt)
    
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
    logging.debug(tuning_results)
    #global prob
    parameters = tuning_results.sort_values(by=['rmse'])
    parameters = parameters.reset_index(drop=True)
    prob = parameters['interval_width'][0]
    logging.debug(parameters)
    
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
        logging.debug("File exists")
    else:
        probs=dict()
        probs[metric] = prob
        logging.debug("File does not exists")
    #writing probabilities in a file
    np.save('prob_file.npy', probs) 
    #logging.debug("file written")
    return final_model
        
def predict(model , number_of_forward_predictions , prediction_horizon , epoch_start):
    #freqInMin = int(prediction_horizon)/60
    ##freq = str(freqInMin) + "min"
    #future = model.make_future_dataframe(periods = number_of_forward_predictions , freq = freq , include_history = False)
    #logging.debug("THIS IS THE FUTURE DATASET")
    #logging.debug(future)
    future = list()
    for i in range(1, number_of_forward_predictions+1):
        dateInSec = epoch_start + i*prediction_horizon
        #logging.debug([dateInSec])
        date=datetime.fromtimestamp(dateInSec)
        future.append(date)
    future = pd.DataFrame(future)
    future.columns = ['ds']
    forecast = model.predict(future)
    logging.debug(forecast)
    return forecast
    
    
    
    
    
    

class Listener(messaging.listener.MorphemicListener):
    
    def __init__(self, m, topic_name):
        self.m = m
        self.m.topic("metrics_to_predict.prophet",id)
        self.topic_name = topic_name

    
    def on_message(self, frame):

          
        logging.debug("Here is the frame")
        logging.debug(frame)
        body=frame.body
        res = ast.literal_eval(body)
        #logging.debug(res[0])
        logging.debug("METRICS TO PREDICT") 
        '''
        #getting data from datasetmaker
        dataset_preprocessor = CSVData(APP_NAME)
        dataset_preprocessor.prepare_csv()
        logging.debug("DATASET DOWNLOADED")
        '''
        logging.debug(res[0])
        
        for r in res:
            metric = r['metric']
        #for metric in metrics:
            if not os.path.isfile('prophet_'+metric+".pkl"): 
                model=train(metric)
                pkl_path = "prophet_"+metric+".pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(model, f)
                logging.debug("TRAINING ENDED FOR: " + metric)
                #flags[metric]=1
            metrics.add(metric)
        #logging.debug("20 minutes")
        
        #model=train("20 minutes")
        
        
        
        logging.debug(metrics)
        
        self.m.send_to_topic("training_models",
            {

            "metrics": list(metrics),

             "forecasting_method": "Prophet",

            "timestamp": int(time())
            }   
        )
        
        

        
        
        
         
         
         
        
class StartForecastingListener(messaging.listener.MorphemicListener):

    def __init__(self, m1, topic_name):
        self.m1 = m1
        self.m1.topic("start_forecasting.prophet",id)
        self.topic_name = topic_name
        
    
        
    
    def on_message(self, frame):
    
            
        body=json.loads(frame.body)
        logging.debug(body)

        sent_metrics = body["metrics"]
        for metric in sent_metrics:
            t = threading.Thread(target=worker , args=(self, frame, metric,))
            t.start()
        
        
        
class StopForecastingListener(messaging.listener.MorphemicListener):

    def __init__(self, m2, topic_name2):
        self.m2 = m2
        self.m2.topic("stop_forecasting.prophet",id)
        self.topic_name2 = topic_name2

    
    def on_message(self, frame):
        logging.debug("STOP FORECASTING")
        
        body=json.loads(frame.body)
        logging.debug("METRICS TO STOP PREDICTION") 
        logging.debug(body["metrics"])
        metrics.add("memory")
        for metric in body["metrics"]:
            if metric in metrics:
                logging.debug("Un-subscribing from: " + metric )
                metrics.remove(metric)
        logging.debug(metrics)
        logging.debug("STOP FORECASTING") 
        


def main():
    logging.getLogger().setLevel(logging.DEBUG)
    
    m = messaging.morphemic.Connection('aaa','111', host='147.102.17.76', port=61610)
    m1 = messaging.morphemic.Connection('aaa','111', host='147.102.17.76', port=61610)
    m2 = messaging.morphemic.Connection('aaa','111', host='147.102.17.76', port=61610)
    
    
    m.connect()
    m1.connect()
    m2.connect()
    
    m.set_listener(id,Listener(m,"/topic/metrics_to_predict.prophet"))
    m1.set_listener(id,StartForecastingListener(m1,"/topic/start_forecasting.prophet"))
    m2.set_listener(id,StopForecastingListener(m2,"/topic/stop_forecasting.prophet"))
    

    while True:
        pass


if __name__ == "__main__":
    publish_rate = 0
    all_metrics = {}

    main()

    
