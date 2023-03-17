import socket
import time
import traceback
import numpy as np
from datetime import datetime
from termcolor import colored
import os

from configs.database import QC4DB_PLUGIN_HOST, QC4DB_PLUGIN_PORT
import src.database.joo_server_utils as jh

class JooServer:

    prefix = "rl_for_jo"
    nextOrder = 0

    def __init__(self) -> None:
        os.makedirs("servers", exist_ok=True)
        timestr = time.strftime("%m-%d_%H-%M-%S")
        self.joinLog = open("servers/" + self.prefix + "_" + timestr + ".log", 'w')

    def onAccept(self, conn, join_order):
        logData = []
        # read data
        data = b'' + conn.recv(1024)
        splitpoint = data.index(0)  # seperator between tables and weights
        while splitpoint < 0:
            data = data + conn.recv(1024)
            splitpoint = data.index(0)
        # parse table names
        namesStr = data[:splitpoint].decode('utf-8')
        tables = namesStr.split(";")
        # read cardinalities
        data = data[splitpoint + 1:]
        number_of_relations = len(tables)
        expectedLen = 8 * ((1 << number_of_relations) - 1 - number_of_relations)
        # read until all data arrived
        while len(data) < expectedLen:
            data = data + conn.recv(1024)
        # parse cardinalities
        cardinalities = np.frombuffer(data, dtype=np.int64).tolist()
        print("Features:")
        print(tables, cardinalities)
        logData.append(tables)
        logData.append(cardinalities)

        logString = ";".join(str(x) for x in logData)
        self.joinLog.write(logString)
        self.joinLog.write("\n")
        self.joinLog.flush()
        print("Log:", colored(logString, 'yellow'))
        print("Order choosen " + str(join_order))

        idString = jh.toIntString(join_order, tables)
        print(idString)
        conn.send(np.array(idString, dtype=np.int16).tobytes())

    def run(self, join_order):
        # open server socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((QC4DB_PLUGIN_HOST, QC4DB_PLUGIN_PORT))
            s.listen()
            print("Server started")
            try:
                print("Listening for connection")
                # wait for connection
                conn, addr = s.accept()
                self.onAccept(conn, join_order)
            except Exception as e:
                print(e)
                traceback.print_exc()
                s.close()

