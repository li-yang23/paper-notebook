# Q-Learning Client
from twisted.internet import reactor,protocol
import libs.grammar.q_protocol as q_protocol
import time
import socket
import argparse
import os
import shutil

import pandas as pd
from libs.caffe.model_exec import ModelExec
from libs.misc.clear_trained_models import clear_redundant_logs_caffe

def start_reactor(clientname,hostname,model,gpu_to_use,debug):
	_model = __import__('models.' + model,
						globals(),
						locals(),
						['hyper_parameters','state_space_parameters'],
						-1)
	if gpu_to_use is not None:
		print('GPU TO USE', gpu_to_use)
		_model.hyper_parameters.CHECKPOINT_DIR = _model.hyper_parameters.CHECKPOINT_DIR+str(gpu_to_use)
	
	f = QFactory(clientname,_model.hyper_parameters,_model.state_space_parameters,gpu_to_use,debug)
	reactor.connectTCP(hostname,8000,f)
	reactor.run()

def main():
	parser = argparse.ArgumentParser()

	model_pkgpath = os.path.join(os.path.dirname(__file__),'models')
	model_choices = next(os.walk(model_pkgpath))[1] # 这个会找到models下面的每一个文件夹

	parser.add_argument('model',
						help='model package name package should have a model.py,' + 
						'file, hyper_parameters.py file, and a log folder',
						choices=model_choices)
	
	parser.add_argument('clientname')
	parser.add_argument('hostname')
	parser.add_argument('-gpu','--gpu_to_use',help='GPU number to use',type=int)
	parser.add_argument('--debug',type=bool,
						help='True if you do not want to actually run networks and return bs',
						default=False)
	
	args = parser.parse_args()

	start_reactor(args.clientname,args.hostname,args.model,args.gpu_to_use,args.debug)

if __name__ == '__main__':
	main()