import messaging
import os
import time
import logging
import signal
import json
from time import time
#from amq_message_python_library import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

id="prophet"
#id2="eshybrid"
#idcnn="prophet"

#m = messaging.morphemic.Connection('aaa','111', host='147.102.17.76', port=61610)

#morphemic = morphemic.model.Model(m)
class Listener(messaging.listener.MorphemicListener):
    def __init__(self):
        self.m = messaging.morphemic.Connection('aaa','111', host='147.102.17.76', port=61610)

 
    def on_training_models_prophet(self, frame):
        global model
        logging.debug("Here is the function")
        body=frame.body
        print(body)
        logging.debug(body)
        logging.debug("TTTTTTTTTT") 
        #sent_metrics = body[0]['metric']
        #prediction_horizon = body["prediction_horizon"]
        #number_of_forward_predictions = body["number_of_forward_predictions"]
        #logging.debug(metrics)
        #horizon=str(prediction_horizon)+' minutes'
        logging.debug("20 minutes")
        #logging.debug(horizon)
        #model=train("20 minutes")   

    '''def on_message(self, frame):
        global model
        logging.debug("Here is the function")
        body=frame.body
        print(body)
        logging.debug(body)
        logging.debug("XxxxX") 
        #sent_metrics = body[0]['metric']
        #prediction_horizon = body["prediction_horizon"]
        #number_of_forward_predictions = body["number_of_forward_predictions"]
        #logging.debug(metrics)
        #horizon=str(prediction_horizon)+' minutes'
        logging.debug("20 minutes")
        #logging.debug(horizon)
        #model=train("20 minutes")    '''  

    def on_start_forecasting_prophet(self, headers):
        logging.debug("I am starting forecasting for Prophet")

    def on_metrics_to_predict(self):
        logging.debug("I am starting forecasting for Prophet")
    def on_metrics_to_predict(self):
        logging.debug("I am starting forecasting for Prophet")
    def on_metrics_to_predict_prophet(self):
        logging.debug("I am starting forecasting for Prophet")
        
    def on_disconnected(self):
        print('disconnected')
        self.m.connect()

    def test(self):
        self.m.set_listener(id,self)
        self.m.connect()  
        print("connected")   
        self.m.topic("start_forecasting_prophet",id)
        self.m.topic("stop_forecasting",id)
        self.m.topic("/topic/metrics_to_predict",id)
        self.m.topic("/topic/training_models_prophet",id)
        print("test")
        self.m.send_to_topic(self, "/topic/training_models_prophet",{"metrics": ["cpu_usage","test"],"forecasting_method": "Prophet","timestamp": int(time()/1000)})   
        print("test2")
#m.set_listener(id,Listener(m,"/topic/metrics_to_predict.prophet"))

l =Listener()
l.test()

while True:
    pass

#m.disconnect()

    
