import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'pcnn_att', help = 'name of the model')
args = parser.parse_args()
model = {
	'pcnn_att': models.PCNN_ATT,
	'pcnn_one': models.PCNN_ONE,
	'pcnn_ave': models.PCNN_AVE,
	'cnn_att': models.CNN_ATT,
	'cnn_one': models.CNN_ONE,
	'cnn_ave': models.CNN_AVE
}
con = config.Config()
con.set_word_size(300)
con.set_data_path('./chinese_data')
con.set_use_bag(False)
con.set_num_classes(35)

# con.set_max_epoch(15)
# con.load_test_data()
# con.set_test_model(model[args.model_name])
# con.set_epoch_range([7,12])
# con.test()

con.load_predict_data()
con.set_test_model(model[args.model_name])
res = con.predict('./checkpoint/PCNN_ATT-14')
with open('./chinese_data/result_sent.txt', 'wt') as f:
	for i, tag in enumerate(res):
		f.write(f'TEST_SENT_ID_{i+1:06}\t{tag}\n')

