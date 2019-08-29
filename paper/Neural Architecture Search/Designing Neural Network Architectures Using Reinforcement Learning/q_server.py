"""
https://github.com/bowenbaker/metaqnn/blob/master/q_server.py
https://blog.csdn.net/suoyan1539/article/details/79571010
https://blog.csdn.net/Allenalex/article/details/78220926
https://blog.csdn.net/qq_30615903/article/details/80739243
https://zhuanlan.zhihu.com/p/35261164
"""
from twisted.internet import reactor,protocol
from twisted.internet.defer import DeferredLock

import libs.grammar.q_protocol as q_protocol
import libs.grammar.q_learner as q_learner

import pandas as pd
import numpy as np

import argparse
import traceback
import os
import socket
import time

class bcolors:
    HEADER = '\033[95m'
    YELLOW = '\033[93m'
    OKBLUE = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class QServer(protocol.ServerFactory):
    def __init__(self,
                 list_path,              # 盲猜一波是保存网络结构的list
                 state_space_parameters, # 状态空间参数，每个状态定义为所有相关层的参数的tuple
                 hyper_parameters,       # 超参数，包含每个神经网络层的相关的参数tuple
                 epsilon=None,           # 反向3，和强化学习步骤相关，因为使用的是epsilon贪心策略
                 number_models=None):
        self.protocol = QConnection
        self.new_net_lock = DeferredLock()
        self.clients = {} # name of connection is key, each value is dict with {'connection','net','iters_sampled'}

        self.replay_columns = ['net',
                               'accuracy_best_val',
                               'iter_best_val',
                               'accuracy_last_val',
                               'iter_last_val',
                               'accuracy_best_test',
                               'accuracy_last_test',
                               'ix_q_value_update',
                               'epsilon',
                               'time_finished',
                               'machine_run_on']
        
        self.list_path = list_path

        self.replay_dictionary_path = os.path.join(list_path,'replay_database.csv')
        self.replay_dictionary,self.q_training_step = self.load_replay()

        self.schedule_or_single = False if epsilon else True
        if self.schedule_or_single:
            self.epsilon = state_space_parameters.epsilon_schedule[0][0]
            self.number_models = state_space_parameters.epsilon_schedule[0][1]
        else:
            self.epsilon = epsilon
            self.number_models = number_models if number_models else 10000000000
        self.state_space_parameters = state_space_parameters
        self.hyper_parameters = hyper_parameters

        self.number_q_updates_per_train = 100

        self.list_path = list_path
        self.qlearner = self.load_qlearner()
        self.check_reached_limit()


    def load_replay(self):
        pass
    
    def load_qlearner(self):
        pass
    
    def check_reached_limit(self):
        pass