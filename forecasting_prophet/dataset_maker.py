from morphemic.dataset import DatasetMaker
import os
from filelock import FileLock

"""Script for preparing csv data downloaded form InfluxDB database, data"""


class CSVData(object):
    def __init__(self, name, start_collection=None):
        self.name = name
        self.config = {
            "hostname": os.environ.get("INFLUXDB_HOSTNAME"),
            "port": int(os.environ.get("INFLUXDB_PORT")),
            "username": os.environ.get("INFLUXDB_USERNAME"),
            "password": os.environ.get("INFLUXDB_PASSWORD"),
            "dbname": os.environ.get("INFLUXDB_DBNAME"),
            "path_dataset": os.environ.get("DATA_PATH"),
        }
        self.start_collection = start_collection

    def prepare_csv(self):
        lockfile = os.path.join(self.config["path_dataset"], f"{self.name}.csv")
        lock = FileLock(lockfile + ".lock")

        if os.path.isfile(lockfile):
            with lock:
                datasetmaker = DatasetMaker(
                    self.name, self.start_collection, self.config
                )
                response = datasetmaker.make()

        else:
            datasetmaker = DatasetMaker(self.name, self.start_collection, self.config)
            response = datasetmaker.make()
