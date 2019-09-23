
import os
from logger import Logger
import json
import time

synced_block = -1

def _block_json(block_no):
    return "./records/block"+str(block_no)+".json"

logger = Logger("./log/")
stepcount = 0
while True:
    if os.path.isfile(_block_json(synced_block+1)):
        time.sleep(2)
        fh = open(_block_json(synced_block+1))
        records = json.load(fh)
        for iloss in range(len(records[0])):
            logger.scalar_summary('loss', records[0][iloss], iloss + stepcount)
        for iaccu in range(len(records[1])):
            logger.scalar_summary('accu', records[1][iaccu], iaccu + stepcount)

        stepcount += len(records[0])
        synced_block += 1
        os.remove("./records/block"+str(synced_block)+".json")

        print("displayed block #{}".format(synced_block))