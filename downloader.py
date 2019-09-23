import socket
import os
import time
import json

HOST = '166.111.121.42'
PORT = '1822'
PWD = "123liaoyuanda"
PROJ = "/Users/liaoyuanda/Desktop/PosNegNet/"
TIME_PER_BLOCK = 10 # expected seconds to generate a block

synced_blocks = -1

def _wait(seconds):
    time.sleep(seconds)

_wait(TIME_PER_BLOCK)
while True:
    try:
        ok = os.system("sshpass -p {} scp -P {} liaoyuanda@{}:~/PosNegNet/records/block{}.json {}/records/".format(\
            PWD, PORT, HOST, synced_blocks+1, PROJ)\
        )
        if ok == 0:
            print("download record block #{} success".format(synced_blocks+1))
            synced_blocks += 1
            if len(json.load(open("./records/block{}.json".format(synced_blocks)))) == 0: break
            _wait(TIME_PER_BLOCK)
        else:
            _wait(2)

    except Exception as e:
        print("error: ", e)
        break