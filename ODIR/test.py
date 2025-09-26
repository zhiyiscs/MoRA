import open_clip
import random
import torch
import io
import pyarrow as pa
import os
import numpy as np
from PIL import Image

train_table = pa.ipc.RecordBatchFileReader(pa.memory_map("/ssd4tb/datasets/chestXray/chestXray_train.arrow", "r")).read_all()
#dev_table = pa.ipc.RecordBatchFileReader(pa.memory_map("/ssd4tb/datasets/chestXray/chestXray_val.arrow", "r")).read_all()
#test_table = pa.ipc.RecordBatchFileReader(pa.memory_map("/ssd4tb/datasets/chestXray/chestXray_test.arrow", "r")).read_all()


print(len(train_table))