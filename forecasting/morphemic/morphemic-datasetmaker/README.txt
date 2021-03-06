1. Generality

Dataset maker is morphemic python library for 
building dataset from data points registered into InfluxDB.
Dataset maker receives the name of an application, the start time 
and the tolerance interval. More details are provided below.

2. InfluxDB format 

Data points in InfluxDB should have the following format for being used
correctly by the dataset maker:

measurement : "application_name" #mandatory 
timestamp : timestamp #optional
fields : dictionnary containing metric exposed by the given application 
         cpu_usage, memory_consumption, response_time, http_latency
tags : dictionnary of metrics related information

The JSON describing the above information is the following:

Ex.: 
    {"measurement": "application_name", 
      "timestamp": 155655476.453, 
      "fields": {
          "cpu_usage": 40,
          "memory_consumption": 67.9,
          "response_time": 28,
          "http_latency": 12
      },
      "tags": {
          "core": 2 #cpu_usage of 40% is the usage of the cpu core number 2
      }
    }
 
If data points are presented as the above format, the dataset maker will output 
a csv (application_name.csv) file with the following schema:
time, cpu_usage, memory_consumption, response_time, http_latency, core

3. Usage 


Warming : make sure the above variables exist before importing dataset make library

from morphemic.dataset import DatasetMaker 

data_maker = DatasetMaker(application, start, configs)
response  = data_maker.make()

application, string containing the application name 
start, when to start building the dataset 
Ex.: '10m' , build dataset containg data point stored the 10 last minute
Ex.: '3h', three hours 
Ex.: '4d', four days 
leave empty or set to None if you wish all data points stored in your InfluxDB
configs is dictionnary containg parameters

{
    "hostname": hostname or IP of InfluxDB
    "port": port of InfluxDB
    "username": InfluxDB username 
    "password": password of the above user 
    "dbname": database name 
    "path_dataset": path where the dataset will be saved
}

the response contains 
{'status': True,'url': url, 'application': application_name, 'features': features}

or if an error occured
{'status': False,'message': "reason of the error"}