import os
from multiprocessing import Pool
import stomp
import json
from amq_message_python_library import *  # python amq-message-python-library
import logging


AMQ_USER = os.environ.get("AMQ_USER", "admin")
AMQ_PASSWORD = os.environ.get("AMQ_PASSWORD", "admin")
START_APP_TOPIC = "metrics_to_predict"
METHOD = os.environ.get("METHOD", "gluonts")
START_TOPIC = f"start_forecasting.{METHOD}"

'''
def run_process(args):
    print("running")
    os.system(f"python {args[0]} '{args[1]}'")


def start_training(metrics_to_predict):
    processes = (("predict.py", metrics_to_predict))
    pool = Pool(processes=1)
    # execute the function run_process on each of the two python files (in processes)
    pool.map(run_process, processes)
    print("I am in start_training")

class StartListener(stomp.ConnectionListener):
    """Custom listener, parameters:
    - conn (stomp connector)
    - topic_name, name of topic to subscribe"""

    def __init__(self, conn, topic_name):
        self.conn = conn
        self.topic_name = topic_name

    def on_error(self, frame):
        print('received an error "%s"' % frame.body)

    def on_message(self, frame):
        print(self.topic_name)
        logging.debug(f" Body: {frame.body}")
        message = json.loads(frame.body)
        global publish_rate, all_metrics
        publish_rate = message[0]["publish_rate"]
        all_metrics = message


class StartForecastingListener(stomp.ConnectionListener):
    """Custom listener, parameters:
    - conn (stomp connector)
    - topic_name, name of topic to subscribe"""

    def __init__(self, conn, topic_name):
        self.conn = conn
        self.topic_name = topic_name

    def on_error(self, frame):
        print('received an error "%s"' % frame.body)

    def on_message(self, frame):
        message = json.loads(frame.body)
        message["publish_rate"] = publish_rate
        message["all_metrics"] = all_metrics
        message = json.dumps(message)
        start_training(message)
        self.conn.disconnect()
'''

def main():
    #logging.getLogger().setLevel(logging.DEBUG)

    start_app_conn = morphemic.Connection(username = "morphemic", password = "morphemic" , host="tcp://147.102.17.76", port=61616)
    start_app_conn.connect()
    #start_conn = morphemic.Connection(AMQ_USER, AMQ_PASSWORD)
    #start_conn.connect()

    #start_conn.conn.subscribe(f"/topic/{START_APP_TOPIC}", "1", ack="auto")
    start_app_conn.conn.subscribe(f"/topic/{START_TOPIC}", "2", ack="auto")

    #start_conn.conn.set_listener("1", StartListener(start_conn.conn, START_APP_TOPIC))
    #start_app_conn.conn.set_listener(
    #    "2", StartForecastingListener(start_conn.conn, START_TOPIC)
    #)

    while True:
        pass


if __name__ == "__main__":
    publish_rate = 0
    all_metrics = {}

    main()
