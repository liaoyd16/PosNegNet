import socket
import json
import time
import numpy as np

def _gen_loss(array_len):
    return (1.5 + np.random.randn(array_len)).tolist()

def _gen_accu(array_len):
    return (70 + 5*np.random.randn(array_len)).tolist()

for i in range(100):
    time.sleep(5)
    print("batch #{} complete".format(i))
    json.dump([_gen_loss(100),_gen_accu(100)], open("./records/block{}.json".format(i), "w"))