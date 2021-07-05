import messaging
import morphemic
from prophet import prophet
from time import time
import logging
import signal
import threading
import numpy as np

# Libraries required for training and prediction
import os
import json
import pickle
import ast
from time import sleep
from dataset_maker import CSVData



APP_NAME = os.environ.get("APP_NAME")
ACTIVEMQ_USER = os.environ.get("ACTIVEMQ_USER")
ACTIVEMQ_PASSWORD = os.environ.get("ACTIVEMQ_PASSWORD")
ACTIVEMQ_HOSTNAME = os.environ.get("ACTIVEMQ_HOSTNAME")
ACTIVEMQ_PORT = os.environ.get("ACTIVEMQ_PORT")

predictionTimes = dict()
models = dict()
flags = {'cpu_usage': 0 , 'latency': 0 , 'memory': 0 ,  'response_time': 0}

def worker(self,body,metric):
    
    logging.debug("Forecasting metric: " + metric)
    timestamp = body['timestamp']
    prediction_horizon = body["prediction_horizon"]
    number_of_forward_predictions = body["number_of_forward_predictions"]   
    logging.debug(metric)
    epoch_start= body["epoch_start"]
    predictionTimes[metric] = epoch_start

    if  os.path.isfile('prophet_'+metric+".pkl"):  
        logging.debug("Loading the trained model for metric: " + metric)

    while(True):
        if flags[metric] == 0:
            epoch_start = predictionTimes[metric]
            flags[metric] = 1
            #load the model
        with open("prophet_"+metric+".pkl", 'rb') as f:
            models[metric] = pickle.load(f)
        
        predictions=prophet.predict(models[metric] , number_of_forward_predictions , prediction_horizon , epoch_start)
        yhats = predictions['yhat'].values.tolist()
        yhat_lowers = predictions['yhat_lower'].values.tolist()
        yhat_uppers = predictions['yhat_upper'].values.tolist()
        
        prediction_time= epoch_start+ prediction_horizon
        timestamp = int(time())
        
        #read probabilities file
        logging.debug("Loading the trained model probabilities")
        probs = np.load('prob_file.npy' , allow_pickle='TRUE').item()

    
        logging.debug("Sending predictions for metric: "+ metric)
        logging.debug("Prediction_time: "+ str(prediction_time))

        for k in range(0,len(predictions['yhat'].values.tolist())):
            yhat = yhats[k]
            yhat_lower = yhat_lowers[k]
            yhat_upper = yhat_uppers[k]
            
            self.connector.send_to_topic('intermediate_prediction_prophet_'+metric,               
            
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


class Prophet(morphemic.handler.ModelHandler,messaging.listener.MorphemicListener):
    id = "prophet"
    metrics = set()
    #probabilities = dict()

    def __init__(self):
        self._run =  False
        logging.debug(ACTIVEMQ_USER)
        logging.debug(ACTIVEMQ_PASSWORD)
        logging.debug(ACTIVEMQ_HOSTNAME)
        logging.debug(ACTIVEMQ_PORT)
        sleep(90)
        logging.debug("slept 90 seconds")
        self.connector = messaging.morphemic.Connection(ACTIVEMQ_USER,ACTIVEMQ_PASSWORD, host=ACTIVEMQ_HOSTNAME, port=ACTIVEMQ_PORT)
        #self.connector = messaging.morphemic.Connection('morphemic','morphemic', host='147.102.17.76', port=61616)
        #self.model = morphemic.model.Model(self)

    def run(self):
        logging.debug("setting up")
        self.connector.connect()
        self.connector.set_listener(self.id, self)
        self.connector.topic("metrics_to_predict_prophet", self.id)
        self.connector.topic("start_forecasting_prophet", self.id)
        self.connector.topic("stop_forecasting_prophet", self.id)

    def reconnect(self):
        print('Reconnecting to ActiveMQ')
        self.connector.disconnect()
        self.run()
        pass

    def on_start_forecasting_prophet(self, body):
        logging.debug("Prophet Start Forecasting the following metrics :") 
        sent_metrics = body["metrics"]
        logging.debug(sent_metrics)
        for metric in sent_metrics:
            if metric not in self.metrics:
                self.metrics.add(metric)
            thread = threading.Thread(target=worker , args=(self, body, metric,))
            thread.start()

    def on_metrics_to_predict_prophet(self, body):
        logging.debug("check the trained model for :") 
        logging.debug(body) 
        #getting data from datasetmaker
        dataset_preprocessor = CSVData(APP_NAME)
        dataset_preprocessor.prepare_csv()
        logging.debug("DATASET DOWNLOADED")
        
        for r in body:
            metric = r['metric']
        #for metric in metrics:
            if not os.path.isfile('prophet_'+metric+".pkl"): 
                logging.debug("Training a Prophet model for metric : " + metric)
                model=prophet.train(metric)
                pkl_path = "prophet_"+metric+".pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(model, f)
                #flags[metric]=1
            self.metrics.add(metric)
        
        self.connector .send_to_topic("training_models",
            {

            "metrics": list(self.metrics),

            "forecasting_method": "Prophet",

            "timestamp": int(time())
            }   
        )

    def on_stop_forecasting_prophet(self, body):
        logging.debug("Prophet Stop Forecasting the following metrics :")
        logging.debug(body["metrics"])
        metrics.add("memory")
        for metric in body["metrics"]:
            if metric in metrics:
                logging.debug("Remove from the list of metrics this metric: " + metric )
                self.metrics.remove(metric)

    def start(self):
        logging.debug("Staring Prophet Forecaster")
        self.run()
        self._run = True 

    def on_disconnected(self):
        print('Disconnected from ActiveMQ')
        self.reconnect()
