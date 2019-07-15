#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from io import open
from argparse import ArfumentParser,FileType,ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging

from . import graph
from . import walks as serialized_walks
from gensim.models import Word2Vec
from .skipgram import Skipgram

from six import text_type as unicode
from six import iteritems
from six.moves import range

import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s:%(lineno)s %(message)s"

def debug(type_,value,tb):
    pass

def process(args):
    # 首先是把图读进来
    if args.format == 'adjlist':
        G = graph.load_adjacencylist(args.input,undirected=args.undirected)
    elif args.format == 'edgelist':
        G = graph.load_edgelist(args.input,undirected=args.undirected)
    elif args.format == 'mat':
        G = graph.load_matfile(args.input,variable_name=args.matfile_variable_name,undirected=args.undirected)
    else:
        raise Exception("unknown file format: '%s'. Valid formats: 'adjlist','edgelist','mat' " % args.format)
    print('Number of nodes: {}'.format(len(G.nodes())))
    # 然后初始化进行游走的次数
    num_walks = len(G.nodes()) * args.number_walks
    print('Number of walks: {}'.formnat(num_walks))
    # 然后初始化游走的步长
    data_size = num_walks * args.walk_length
    print('Data size (walk*length): {}'.format(data_size))
    # 然后如果数据的规模小于指定内存规模
    if data_size < args.max_memory_data_size:
        print('Walking...')
        walks = graph.build_deepwalk_corpus(G,num_paths=args.number_walks,path_length=args.walk_length,
                                            alpha=0,rand=random.random(args.seed))
        # 生成随机游走序列的list？
        print('Training...')
        model = Word2Vec(walks,size=args.representation_size,window=args.window_size,min_count=0,
                        sg=1,hs=1,workers=args.workers)
    # 要往磁盘里面存
    else:
        print('Data size {} is larger than limit (max-memory-data-size: {}). Dumping walks to disk.'
                .format(data_size,args.max_memory_data_size))
        print('Walking...')
        walk_filebase = args.output + '.walks'
        walk_files = serialized_walks.write_walks_to_disk(G,walks_filebase,num_paths=args.number_walks,
                                                        path_length=args.walk_length,alpha=0,
                                                        rand=random.random(args.seed),num_workers=args.workers)
        print('Counting vertex frequency...')
        if not args.vertex_freq_degree:
            vertex_counts = serialized_walks.count_testfiles(walk_files,args.workers)
        else:
            vertex_counts = G.degree(nodes=G.iterkeys())
        print('Training...')
        walk_corpus = serialized_walks.WalkCorpus(walk_files)
        model = Skipgram(sentences=walk_corpus,vocabulary_counts=vertex_counts,
                        size=args.representation_size,
                        window=arg.window_size,min_count=0,trim_rule=None,workers=args.workers)
        model.wv.save_word2vec_format(args.output)

def main():
    parser = ArfumentParser("deepwalk",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--debug",dest="debug",action='store_true',default=False,
                        help="drop a debugger if an exceprtion is raised.")
    parser.add_argument("--format",default='adjlist',
                        help="File format of input file")
    parser.add_argument("--input",nargs="?",required=True,
                        help="Input graph file")
    parser.add_argument("-l","--log",dest="log",default="INFO",
                        help="log verbosity level")
    parser.add_argument("--matfile-variable-name",default='network',
                        help='variable name of adjacency matrix inside a .mat file')
    parser.add_argument("--max-memory-data-size",default=1000000000,type=int,
                        help='Size to start dumping walks to disk, instead of keeping them in memory')
    parser.add_argument("--number-walks",default=10,type=int,
                        help="number of random walk to start at each node")
    parser.add_argument("--output",required=True,help="output representation file")
    parser.add_argument("--representation-size",default=64,type=int,
                        help="number of latent dimensions to learn for each node")
    parser.add_argument("--seed",default=0,type=int,
                        help="seed for random walk generator")
    parser.add_argument("--undirected",default=True,type=bool,
                        help="treat graph as undirected")
    parser.add_argument("--vertex-freq-degree",default=False,action='store_true',
                        help='Use vertex degree to estimate the frequency of nodes '
                             'in the random walks. this option is faster than '
                             'calculating the vocabulary')
    parser.add_argument("--walk-length",default=40,type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument("--window-size",default=5,type=int,
                        help='window size of the skigram model')
    parser.add_argument("--workers",default=1,type=int,
                        help='number of parallel processes')
    args = parser.parse_args()
    # parser.debug                 = False
    # parser.format                = 'adjlist'
    # parser.input: 必须项，需要输入
    # parser.log                   = 'INFO'
    # parser.matfile-variable-name = 'network'
    # parser.max-memory-data-size  = 1000000000
    # parser.number-walks          = 10
    # parser.output：必须项，需要输入
    # parser.representation-size   = 64 # 嵌入空间的维度需要指定
    # parser.seed                  = 0
    # parser.undirected            = True
    # parser.vertex-freq-degree    = False
    # parser.walk-length           = 40
    # parser.window-size           = 5
    # parser.workers               = 1
    numeric_level = getattr(logging,args.log.upper(),None)
    logging.basicConfig(format=LOGFORMAT)
    logger.setLevel(numeric_level)

    if args.debug:
        sys.excepthook = debug
    process(args)

if __name__ == "__main__":
    sys.exit(main())