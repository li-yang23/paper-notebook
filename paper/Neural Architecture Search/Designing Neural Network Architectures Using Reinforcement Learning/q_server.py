"""
Q-Learning server
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
                 list_path,              # 盲猜一波是保存网络结构的list？
                 state_space_parameters, # 状态空间参数，每个状态定义为所有相关层的参数的tuple
                 hyper_parameters,       # 超参数，包含每个神经网络层的相关的参数tuple
                 epsilon=None,           # 反向3，和强化学习步骤相关，因为使用的是epsilon贪心策略
                 number_models=None):

        self.protocol = QConnection
        self.new_net_lock = DeferredLock()
        self.clients = {} # name of connection is key, each value is dict with {'connection','net','iters_sampled'}

        # experimential replay策略，将已经训练过的网络保存起来，
        self.replay_columns = ['net',                # net string
                               'accuracy_best_val',  # 验证集的最优的精确值
                               'iter_best_val',      # 验证集的最优的迭代次数
                               'accuracy_last_val',  # 验证集的最差精确值
                               'iter_last_val',      # 验证集的最后一个模型的迭代次数
                               'accuracy_best_test', # 测试集的最优精确值
                               'accuracy_last_test', # 测试集的最差精确值
                               'ix_q_value_update',  # 更新q值的次数
                               'epsilon',            # 给ε贪心策略的参数
                               'time_finished',      # unix time?
                               'machine_run_on']
        
        self.list_path = list_path # list_path = './test/needed_for_testing'

        self.replay_dictionary_path = os.path.join(list_path,'replay_database.csv')
        self.replay_dictionary,self.q_training_step = self.load_replay() # 读取经验存储

        self.schedule_or_single = False if epsilon else True   # 是否要使用多个ε值得策略
        if self.schedule_or_single:
            self.epsilon = state_space_parameters.epsilon_schedule[0][0]  # ε贪心策略的参数，=0时表示完全确定的策略，越接近1说明越随机
            self.number_models = state_space_parameters.epsilon_schedule[0][1]  # 计划生成的模型数目
        else:
            self.epsilon = epsilon  # ε贪心策略的参数，=0时表示完全确定的策略，越接近1说明越随机
            self.number_models = number_models if number_models else 10000000000 # 计划生成的模型数目
        self.state_space_parameters = state_space_parameters
        self.hyper_parameters = hyper_parameters

        self.number_q_updates_per_train = 100

        self.list_path = list_path
        self.qlearner = self.load_qlearner()
        self.check_reached_limit()


    def load_replay(self):
        if os.path.isfile(self.replay_dictionary_path):
            print('Found replay dictionary')
            replay_dic = pd.read_csv(self.replay_dictionary_path)
            q_training_step = max(replay_dic.ix_q_value_update)
        else:
            replay_dic = pd.DataFrame(columns=self.replay_columns)
            q_training_step = 0
        return replay_dic,q_training_step
    
    def load_qlearner(self):
        # Load previous q_values
        # list_path = './test/needed_for_testing/'
        if os.path.isfile(os.path.join(self.list_path,'q_values.csv')): # 这个是q-table？
            print('Found q values')
            qstore = q_learner.QValues()
            qstore.load_q_values(os.path.join(self.list_path,'q_values.csv'))
        else:
            qstore = None
        
        ql = q_learner.QLearner(self.state_space_parameters,
                                self.epsilon,
                                qstore=qstore,
                                replay_dictionary=self.replay_dictionary)
        return ql
    
    def check_reached_limit(self):
        ''' Returns True if the experiment is complete
        '''
        if len(self.replay_dictionary):
            completed_current = self.number_trained_unique(self.epsilon) >= self.number_models
            if completed_current:
                if self.schedule_or_single:
                    # Loop through epsilon schedule, If we find an epsilon that isn't trained, start using that.
                    completed_experiment = True
                    for epsilon, num_models in self.state_space_parameters.epsilon_schedule:
                        if self.number_trained_unique(epsilon) < num_models:
                            self.epsilon = epsilon
                            self.number_models = num_models
                            self.qlearner = self.load_qlearner()
                            completed_experiment = False
                            break
                else:
                    completed_experiment = True
                return completed_experiment
            else:
                return False

    def number_trained_unique(self,epsilon=None):
        '''Epsilon defaults to the minimum'''
        replay_unique = self.filter_replay_for_first_run(self.replay_dictionary)
        eps = epsilon if epsilon else min(replay_unique.epsilon.values)
        replay_unique = replay_unique[replay_unique.epsilon == eps]
        return len(replay_unique)
    
    def filter_replay_for_first_run(self,replay):
        ''' Order replay by iteration, then remove duplicate nets keeping the first'''
        ''' 迭代排序replay，然后去除重复的网络，保留第一个'''
        temp = replay.sort_values(['ix_q_value_update']).reset_index(drop=True).copy()
        return temp.drop_duplicates(['net'])
    
class QConnection(protocol.Protocol):
    def __init__(self):
        pass
    
def main():
    parser = argparse.ArgumentParser()

    model_pkgpath = os.path.join(os.path.dirname(__file__),'models')
    model_choices = next(os.walk(model_pkgpath))[1]

    parser.add_argument('model',
                        help='model package name package should have a model.py,'+
                        'file, hyper_patameters.py file, and a log folder',
                        choices=model_choices)
    parser.add_argument('list_path')
    parser.add_argument('-eps','--epsilon',help='For Epsilon Greedy Strategy',type=float)
    parser.add_argument('-nmt','--number_models_to_train',type=int,
                        help='How many models for this epsilon do you want to train.')
    
    args = parser.parse_args()

    _model = __import__('models.'+args.model,
                        globals(),
                        locals(),
                        ['state_space_paramters','hyper_parameters'],
                        -1)

    factory = QServer(args.list_path,
                    _model.state_space_parameters,
                    _model.hyper_parameters,
                    args.epsilon,
                    args.number_models_to_train)
    
    reactor.listenTCP(8000,factory)
    reactor.run()
    

    pass

if __name__ == '__main__':
    main()