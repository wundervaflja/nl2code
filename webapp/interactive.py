#! /usr/bin/env python

import numpy as np
import cProfile
import ast
import traceback
import argparse
import os
import logging
# from vprof import profiler

# from model import Model
# from dataset import DataEntry, DataSet, Vocab, Action
# import config
# from learner import Learner
# from evaluation import *
# from decoder import decode_python_dataset
# from components import Hyp
# from astnode import ASTNode

# from nn.utils.generic_utils import init_logging
# from nn.utils.io_utils import deserialize_from_file, serialize_to_file

# from dataset import canonicalize_query, query_to_data
# from collections import namedtuple
# from lang.py.parse import decode_tree_to_python_ast

from flask import Flask, render_template
from flask import request
import json


config = {'ptrnet_hidden_dim': 50, 'target_vocab_size': 2101, 'head_nt_constraint': True,
          'random_seed': 181783, 'operation': 'interactive', 'optimizer': 'adam', 'save_per_batch': 4000,
          'rule_embed_dim': 128, 'frontier_node_type_feed': True, 'clip_grad': 0.0, 'valid_metric': 'bleu', 
          'max_epoch': 50, 'max_query_length': 70, 'parent_hidden_state_feed': True, 'word_embed_dim': 128, 
          'source_vocab_size': 2490, 'node_num': 96, 'tree_attention': False, 'output_dir': 'runs', 
          'decode_max_time_step': 100, 'data_type': 'django', 'dropout': 0.2, 'encoder_hidden_dim': 256, 
          'parent_action_feed': True, 'encoder': 'bilstm', 'rule_num': 222, 'batch_size': 10, 'ifttt_test_split': 
          'data/ifff.test_data.gold.id', 'attention_hidden_dim': 50, 'node_embed_dim': 64, 
          'data': 'data/django.cleaned.dataset.freq5.par_info.refact.space_only.bin', 'valid_per_batch': 4000, 
          'enable_copy': True, 'decoder_hidden_dim': 256, 'beam_size': 15, 'mode': 'new', 
          'model': 'models/model.django_word128_encoder256_rule128_node64.beam15.adam.simple_trans.no_unary_closure.8e39832.run3.best_acc.npz', 
          'train_patience': 10}





# def start():
#     if not os.path.exists(config['output_dir']):
#         os.makedirs(config['output_dir'])

#     np.random.seed(config['random_seed'])
#     train_data, dev_data, test_data = deserialize_from_file(config['data'])

#     config_module = sys.modules['config']
#     for name, value in config.iteritems():
#         setattr(config_module, name, value)
#     model = Model()
#     model.build()

#     if config['model']:
#         model.load(config['model'])
#     return model, train_data 

# def interactive(cmd, model, train_data):

#     # we play with new examples!
#     query, str_map = canonicalize_query(cmd)
#     vocab = train_data.annot_vocab
#     query_tokens = query.split(' ')
#     query_tokens_data = [query_to_data(query, vocab)]
#     example = namedtuple('example', ['query', 'data'])(query=query_tokens, 
#                                                        data=query_tokens_data)

#     if hasattr(example, 'parse_tree'):
#         print 'gold parse tree:'
#         print example.parse_tree

#     cand_list = model.decode(example, train_data.grammar, train_data.terminal_vocab,
#                              beam_size=config['beam_size'],
#                              max_time_step=config['decode_max_time_step'])

#     has_grammar_error = any([c for c in cand_list if c.has_grammar_error])
#     print 'has_grammar_error: ', has_grammar_error
#     results = []
#     for cid, cand in enumerate(cand_list[:5]):
#         res = {}
#         res['score'] = cand.score
#         try:
#             ast_tree = decode_tree_to_python_ast(cand.tree)
#             code = astor.to_source(ast_tree)
#             res['code'] = code
#         except:
#             pass
#         finally:
#             res['tree'] = cand.tree.__repr__()
#         results.append(res)
#     return results


app = Flask(__name__)

# app.config['model'], app.config['train'] = start()


@app.route('/', methods=['GET', 'POST'])
def text2code_page():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        edited_code = request.form.get('edited_code')
        if edited_code:
            print edited_code
        text = request.form.get('text')
        if text:
            # res = interactive(text, app.config['model'], app.config['train'])
            # res = sorted(res, key=lambda k: k['score'], reverse=True)
            return render_template('index.html', code=[{'code': 'shot', 'tree': 'shit'}])
        return render_template('index.html', text=text, code=[{'code': 'shot', 'tree': 'shit'}])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)
