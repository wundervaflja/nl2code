#! /usr/bin/env python

import os
import logging

import numpy as np

from flask import Flask, render_template
from flask import request


from nl2code import config

from nl2code.astnode import ASTNode
from collections import namedtuple
from nl2code.components import Hyp
from nl2code.dataset import canonicalize_query, query_to_data
from nl2code.dataset import DataSet, Vocab, Action, DataEntry
from nl2code.decoder import decode_python_dataset
from nl2code.evaluation import *
from nl2code.lang.py.parse import decode_tree_to_python_ast
from nl2code.model import Model
# from nl2code.nn.utils.generic_utils import init_logging
from nl2code.nn.utils.io_utils import deserialize_from_file, serialize_to_file


path = os.path.dirname(os.path.realpath(__file__))

CONFIG = {'ptrnet_hidden_dim': 50,
          'target_vocab_size': 2101,
          'head_nt_constraint': True,
          'random_seed': 181783,
          'operation': 'interactive',
          'optimizer': 'adam',
          'save_per_batch': 4000, 
          'rule_embed_dim': 128, 
          'frontier_node_type_feed': True,
          'clip_grad': 0.0, 
          'valid_metric': 'bleu', 
          'max_query_length': 70,
          'max_epoch': 50,
          'parent_hidden_state_feed': True, 
          'word_embed_dim': 128, 
          'source_vocab_size': 2490,
          'node_num': 96, 'tree_attention': False, 'output_dir': 'runs',
          'decode_max_time_step': 100, 'data_type': 'django', 'dropout': 0.2,
          'encoder_hidden_dim': 256, 'parent_action_feed': True, 'encoder': 'bilstm',
          'rule_num': 222, 'batch_size': 10, 'ifttt_test_split': 'data/ifff.test_data.gold.id',
          'attention_hidden_dim': 50, 'node_embed_dim': 64,  
          'data': '%s/../nl2code/data/django.cleaned.dataset.freq3.par_info.refact.space_only.order_by_ulink_len.bin' % path,
          'valid_per_batch': 4000, 'enable_copy': True, 'decoder_hidden_dim': 256,
          'beam_size': 15, 'mode': 'new', 
          'model': '%s/../nl2code/runs/model.best_acc.npz' % path, 
          'train_patience': 10}


def get_logger():
    logger = logging.getLogger('user_feedback')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    hdlr = logging.FileHandler('./user_feedback.log')
    hdlr.setFormatter(formatter)
    chdlr = logging.StreamHandler()
    chdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.addHandler(chdlr)
    return logger

LOGGER = get_logger()


def start():
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])

    np.random.seed(CONFIG['random_seed'])

    train_data, dev_data, test_data = deserialize_from_file(CONFIG['data'])
    CONFIG['source_vocab_size'] = train_data.annot_vocab.size
    CONFIG['target_vocab_size'] = train_data.terminal_vocab.size
    CONFIG['rule_num'] = len(train_data.grammar.rules)
    CONFIG['node_num'] = len(train_data.grammar.node_type_to_id)
    config_module = sys.modules['nl2code.config']
    for name, value in CONFIG.iteritems():
        setattr(config_module, name, value)
    model = Model()
    model.build()
    if CONFIG['model']:
        model.load(CONFIG['model'])
    return model, train_data 


def interactive(cmd, model, train_data):

    # we play with new examples!
    query, str_map = canonicalize_query(cmd)
    vocab = train_data.annot_vocab
    query_tokens = query.split(' ')
    query_tokens_data = [query_to_data(query, vocab)]
    example = namedtuple('example', ['query', 'data'])(query=query_tokens,
                                                       data=query_tokens_data)
    cand_list = model.decode(example, train_data.grammar, 
                             train_data.terminal_vocab, 
                             beam_size=CONFIG['beam_size'],
                             max_time_step=CONFIG['decode_max_time_step'])

    results = []
    for _, cand in enumerate(cand_list[:5]):
        res = {}
        res['score'] = cand.score
        try:
            ast_tree = decode_tree_to_python_ast(cand.tree)
            code = astor.to_source(ast_tree)
            res['code'] = code
        except:
            pass
        finally:
            res['tree'] = cand.tree.__repr__()
        results.append(res)
    return results


app = Flask(__name__)

app.config['model'], app.config['train'] = start()


@app.route('/', methods=['GET', 'POST'])
def text2code_page():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        edited_code = request.form.get('edited_code')
        text = request.form.get('text')
        if edited_code:
            LOGGER.info(edited_code)
            return render_template('index.html', edited_code=edited_code)
        if text:
            res = []
            try:
                res = interactive(text, app.config['model'], app.config['train'])
                res = sorted(res, key=lambda k: k['score'], reverse=True)
                return render_template('index.html', code=res, text=text)
            except:
                app.config['model'], app.config['train'] = start()
                res = interactive(text, app.config['model'], app.config['train'])
                res = sorted(res, key=lambda k: k['score'], reverse=True)
                return render_template('index.html', code=res, text=text)
            else:
                return render_template('index.html', code=None, text=text)
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

