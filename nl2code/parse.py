import ast
import re
import sys, inspect
from StringIO import StringIO

import astor
from collections import OrderedDict
from tokenize import generate_tokens, tokenize
import token as tk

from nl2code.nn.utils.io_utils import deserialize_from_file, serialize_to_file

from nl2code.tree import *
from nl2code.grammar import *

p_elif = re.compile(r'^elif\s?')
p_else = re.compile(r'^else\s?')
p_try = re.compile(r'^try\s?')
p_except = re.compile(r'^except\s?')
p_finally = re.compile(r'^finally\s?')
p_decorator = re.compile(r'^@.*')

ast_node_black_list = {'ctx'}

valid_ast_leaf_nodes = {ast.Ellipsis, ast.And, ast.Or, ast.Add, ast.Sub}

ast_class_fields = {
    'FunctionDef': {
        'name': {
            'type': 'identifier',
            'is_list': False,
            'is_optional': False
        },
        'args': {
            'type': ast.arguments,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'decorator_list': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        }
    },
    'ClassDef': {
        'name': {
            'type': ast.arguments,
            'is_list': False,
            'is_optional': False
        },
        'bases': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'decorator_list': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        }
    },
    'Return': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
    },
    'Delete': {
        'targets': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'Assign': {
        'targets': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        }
    },
    'AugAssign': {
        'target': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'op': {
            'type': ast.operator,
            'is_list': False,
            'is_optional': False
        },
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        }
    },
    'Print': {
        'dest': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'values': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'nl': {
            'type': bool,
            'is_list': False,
            'is_optional': False
        }
    },
    'For': {
        'target': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'iter': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'orelse': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        }
    },
    'While': {
        'test': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'orelse': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
    },
    'If': {
        'test': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'orelse': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
    },
    'With': {
        'context_expr': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'optional_vars': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
    },
    'Raise': {
        'type': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'inst': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'tback': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
    },
    'TryExcept': {
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'handlers': {
            'type': ast.excepthandler,
            'is_list': True,
            'is_optional': False
        },
        'orelse': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
    },
    'TryFinally': {
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        },
        'finalbody': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        }
    },
    'Assert': {
        'test': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'msg': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'Import': {
        'names': {
            'type': ast.alias,
            'is_list': True,
            'is_optional': False
        }
    },
    'ImportFrom': {
        'module': {
            'type': 'identifier',
            'is_list': False,
            'is_optional': True
        },
        'names': {
            'type': ast.alias,
            'is_list': True,
            'is_optional': False
        },
        'level': {
            'type': int,
            'is_list': False,
            'is_optional': True
        }
    },
    'Exec': {
        'body': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'globals': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'locals': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
    },
    'Global': {
        'names': {
            'type': 'identifier',
            'is_list': True,
            'is_optional': False
        },
    },
    'Expr': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'BoolOp': {
        'op': {
            'type': ast.boolop,
            'is_list': False,
            'is_optional': False
        },
        'values': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'BinOp': {
        'left': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'op': {
            'type': ast.operator,
            'is_list': False,
            'is_optional': False
        },
        'right': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'UnaryOp': {
        'op': {
            'type': ast.unaryop,
            'is_list': False,
            'is_optional': False
        },
        'operand': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'Lambda': {
        'args': {
            'type': ast.arguments,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'IfExp': {
        'test': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'body': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'orelse': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
    },
    'Dict': {
        'keys': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'values': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'Set': {
        'elts': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'ListComp': {
        'elt': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'generators': {
            'type': ast.comprehension,
            'is_list': True,
            'is_optional': False
        },
    },
    'SetComp': {
        'elt': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'generators': {
            'type': ast.comprehension,
            'is_list': True,
            'is_optional': False
        },
    },
    'DictComp': {
        'key': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'generators': {
            'type': ast.comprehension,
            'is_list': True,
            'is_optional': False
        },
    },
    'GeneratorExp': {
        'elt': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'generators': {
            'type': ast.comprehension,
            'is_list': True,
            'is_optional': False
        },
    },
    'Yield': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'Compare': {
        'left': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'ops': {
            'type': ast.cmpop,
            'is_list': True,
            'is_optional': False
        },
        'comparators': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'Call': {
        'func': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'args': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'keywords': {
            'type': ast.keyword,
            'is_list': True,
            'is_optional': False
        },
        'starargs': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'kwargs': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
    },
    'Repr': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        }
    },
    'Num': {
        'n': {
            'type': object,
            'is_list': False,
            'is_optional': False
        }
    },
    'Str': {
        's': {
            'type': str,
            'is_list': False,
            'is_optional': False
        }
    },
    'Attribute': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'attr': {
            'type': 'identifier',
            'is_list': False,
            'is_optional': False
        },
        'ctx': {
            'type': ast.expr_context,
            'is_list': False,
            'is_optional': False
        },
    },
    'Subscript': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'slice': {
            'type': ast.slice,
            'is_list': False,
            'is_optional': False
        },
    },
    'Name': {
        'id': {
            'type': 'identifier',
            'is_list': False,
            'is_optional': False
        }
    },
    'List': {
        'elts': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'ctx': {
            'type': ast.expr_context,
            'is_list': False,
            'is_optional': False
        },
    },
    'Tuple': {
        'elts': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'ctx': {
            'type': ast.expr_context,
            'is_list': False,
            'is_optional': False
        },
    },
    'ExceptHandler': {
        'type': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'name': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'body': {
            'type': ast.stmt,
            'is_list': True,
            'is_optional': False
        }
    },
    'arguments': {
        'args': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
        'vararg': {
            'type': 'identifier',
            'is_list': False,
            'is_optional': True
        },
        'kwarg': {
            'type': 'identifier',
            'is_list': False,
            'is_optional': True
        },
        'defaults': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'comprehension': {
        'target': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'iter': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        },
        'ifs': {
            'type': ast.expr,
            'is_list': True,
            'is_optional': False
        },
    },
    'keyword': {
        'arg': {
            'type': 'identifier',
            'is_list': False,
            'is_optional': False
        },
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'alias': {
        'name': {
            'type': 'identifier',
            'is_list': False,
            'is_optional': False
        },
        'asname': {
            'type': 'identifier',
            'is_list': False,
            'is_optional': True
        }
    },
    'Slice': {
        'lower': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'upper': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        },
        'step': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': True
        }
    },
    'ExtSlice': {
        'dims': {
            'type': ast.slice,
            'is_list': True,
            'is_optional': False
        }
    },
    'Index': {
        'value': {
            'type': ast.expr,
            'is_list': False,
            'is_optional': False
        }
    }
}


def escape(text):
    text = text \
        .replace('"', '-``-') \
        .replace('\'', '-`-') \
        .replace(' ', '-SP-') \
        .replace('\t', '-TAB-') \
        .replace('\n', '-NL-') \
        .replace('(', '-LRB-') \
        .replace(')', '-RRB-') \
        .replace('|', '-BAR-')
    return repr(text)[1:-1] if text else '-NONE-'


def unescape(text):
    text = text \
        .replace('-``-', '"') \
        .replace('-`-', '\'') \
        .replace('-SP-', ' ') \
        .replace('-TAB-', '\\t') \
        .replace('-NL-', '\\n') \
        .replace('-LRB-', '(') \
        .replace('-RRB-', ')') \
        .replace('-BAR-', '|') \
        .replace('-NONE-', '')

    return text


def is_compositional_leaf(node):
    is_leaf = True

    for field_name, field_value in ast.iter_fields(node):
        if field_name in ast_node_black_list:
            continue

        if field_value is None:
            is_leaf &= True
        elif isinstance(field_value, list) and len(field_value) == 0:
            is_leaf &= True
        else:
            is_leaf &= False

    return is_leaf


def ast_to_tree(node):
    if isinstance(node, str):
        label = escape(node)
        return Tree(str, label, holds_value=True)
    elif isinstance(node, int):
        return Tree(int, node, holds_value=True)
    elif isinstance(node, float):
        return Tree(float, node, holds_value=True)

    assert isinstance(node, ast.AST)

    node_type = type(node)
    tree = Tree(node_type)

    # it's a leaf AST node
    if len(node._fields) == 0:
        return tree

    # if it's a compositional AST node with empty fields
    if is_compositional_leaf(node):
        epsilon = Tree('epsilon')
        tree.add_child(epsilon)

        return tree

    # it's a compositional AST node
    fields_info = ast_class_fields[node_type.__name__]

    for field_name, field_value in ast.iter_fields(node):
        # remove ctx stuff
        if field_name in ast_node_black_list:
            continue

        field_type = fields_info[field_name]['type']
        is_list = fields_info[field_name]['is_list']
        is_optional = fields_info[field_name]['is_optional']

        # omit empty fields
        if field_value is None or (is_list and len(field_value) == 0):
            continue

        if isinstance(field_value, ast.AST):
            child = ast_to_tree(field_value)
            tree.add_child(Tree(field_type, field_name, child))
        elif isinstance(field_value, str) or isinstance(field_value, int) or isinstance(field_value, float):
            child = Tree(type(field_value), field_name, ast_to_tree(field_value))
            tree.add_child(child)
        elif is_list:
            if len(field_value) > 0:
                child = Tree(typename(field_type) + '*', field_name)
                # list_node = Tree('list')
                # child.children.append(list_node)
                for n in field_value:
                    if field_type in {ast.comprehension, ast.excepthandler, ast.arguments, ast.keyword, ast.alias}:
                        child.add_child(ast_to_tree(n))
                    else:
                        intermediate_node = Tree(field_type)
                        intermediate_node.add_child(ast_to_tree(n))
                        child.add_child(intermediate_node)

                tree.add_child(child)
        else:
            raise RuntimeError('unknown AST node field!')

    return tree


def parse(code):
    """
    parse a python code into a tree structure
    code -> AST tree -> AST tree to internal tree structure
    """
    root_node = code_to_ast(code)

    tree = ast_to_tree(root_node.body[0])

    tree = add_root(tree)

    return tree


def code_to_ast(code):
    if p_elif.match(code):
        code = 'if True: pass\n' + code

    if p_else.match(code):
        code = 'if True: pass\n' + code

    if p_try.match(code):
        code = code + 'pass\nexcept: pass'
    elif p_except.match(code):
        code = 'try: pass\n' + code
    elif p_finally.match(code):
        code = 'try: pass\n' + code

    if p_decorator.match(code):
        code = code + '\ndef dummy(): pass'

    if code[-1] == ':':
        code = code + 'pass'

    ast_tree = ast.parse(code)

    return ast_tree


def parse_django(code_file):
    line_num = 0
    error_num = 0
    parse_trees = []
    for line in open(code_file):
        code = line.strip()
        # try:
        parse_tree = parse(code)

        leaves = parse_tree.get_leaves()
        for leaf in leaves:
            if not is_terminal_type(leaf.type):
                print parse_tree

        # parse_tree = add_root(parse_tree)

        parse_trees.append(parse_tree)

        # check rules
        # rule_list = parse_tree.get_rule_list(include_leaf=True)
        # for rule in rule_list:
        #     if rule.parent.type == int and rule.children[0].type == int:
        #         # rule.parent.type == str and rule.children[0].type == str:
        #         pass

        # ast_tree = tree_to_ast(parse_tree)
        # print astor.to_source(ast_tree)
            # print parse_tree
        # except Exception as e:
        #     error_num += 1
        #     #pass
        #     #print e

        line_num += 1

    print 'total line of code: %d' % line_num
    print 'error num: %d' % error_num

    assert error_num == 0

    grammar = get_grammar(parse_trees)

    with open('grammar.txt', 'w') as f:
        for rule in grammar:
            str = rule.__repr__()
            f.write(str + '\n')

    with open('parse_trees.txt', 'w') as f:
        for tree in parse_trees:
            f.write(tree.__repr__() + '\n')

    return grammar, parse_trees


def tree_to_ast(tree):
    node_type = tree.type
    node_label = tree.label

    # remove root
    if node_type == 'root':
        return tree_to_ast(tree.children[0])

    if tree.is_leaf and is_builtin_type(node_type):
        return node_label

    else:
        ast_node = node_type()
        node_type_name = typename(node_type)

        # it's a compositional AST node
        if node_type_name in ast_class_fields:
            fields_info = ast_class_fields[node_type_name]

            for child_node in tree.children:
                # if it's a compositional leaf
                if child_node.type == 'epsilon':
                    continue

                field_type = child_node.type
                field_label = child_node.label
                field_entry = ast_class_fields[node_type_name][field_label]
                is_list = field_entry['is_list']

                if is_list:
                    field_type = field_entry['type']
                    field_value = []

                    if field_type in {ast.comprehension, ast.excepthandler, ast.arguments, ast.keyword, ast.alias}:
                        nodes_in_list = child_node.children
                        for sub_node in nodes_in_list:
                            sub_node_ast = tree_to_ast(sub_node)
                            field_value.append(sub_node_ast)
                    else:  # expr stuffs
                        inter_nodes = child_node.children
                        for inter_node in inter_nodes:
                            assert len(inter_node.children) == 1
                            sub_node_ast = tree_to_ast(inter_node.children[0])
                            field_value.append(sub_node_ast)
                else:
                    assert len(child_node.children) == 1
                    sub_node = child_node.children[0]

                    field_value = tree_to_ast(sub_node)

                setattr(ast_node, field_label, field_value)

        for field in ast_node._fields:
            if not hasattr(ast_node, field) and not field in ast_node_black_list:
                if fields_info and fields_info[field]['is_list'] and not fields_info[field]['is_optional']:
                    setattr(ast_node, field, list())
                else:
                    setattr(ast_node, field, None)

        return ast_node


def decode_tree_to_ast(decode_tree):
    decode_tree = decode_tree.children[0]
    terminals = decode_tree.get_leaves()

    for terminal in terminals:
        if type(terminal.label) == str:
            if terminal.label.endswith('<eos>'):
                terminal.label = terminal.label[:-5]
        if terminal.type in {int, float, str}:
            # cast to target data type
            terminal.label = terminal.type(terminal.label)

    ast_tree = tree_to_ast(decode_tree)

    return ast_tree


def tokenize(code):
    token_stream = generate_tokens(StringIO(code).readline)
    tokens = []
    for toknum, tokval, (srow, scol), (erow, ecol), _ in token_stream:
        if toknum == tk.ENDMARKER:
            break
        tokens.append(tokval)

    return tokens


if __name__ == '__main__':
    #     node = ast.parse('''
    # # for i in range(1, 100):
    # #  sum = sum + i
    # #
    # # sorted(arr, reverse=True)
    # # sorted(my_dict, key=lambda x: my_dict[x], reverse=True)
    # # m = dict ( zip ( new_keys , keys ) )
    # # for f in sorted ( os . listdir ( self . path ) ) :
    # #     pass
    # for f in sorted ( os . listdir ( self . path ) ) : pass
    # ''')
    # print ast.dump(node, annotate_fields=False)
    # print get_tree_str_repr(node)
    # print parse('sorted(my_dict, key=lambda x: my_dict[x], reverse=True)')
    # print parse('global _standard_context_processors')

    # parse_django('/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code')

    # code = 'sum = True'
    code = """sorted(my_dict, key=lambda x: my_dict[x], reverse=True)"""
    # # # # gold_ast_node = code_to_ast(code)
    parse_tree = parse(code)
    parse_tree = add_root(parse_tree)
    rule_list = parse_tree.get_rule_list()
    # print parse_tree
    # ast_tree = tree_to_ast(parse_tree)
    # # # # #
    # import astor
    # print astor.to_source(ast_tree)

    from dataset import DataSet, Vocab, DataEntry, Action
    # train_data, dev_data, test_data = deserialize_from_file('django.cleaned.dataset.bin')
    # cand_list = deserialize_from_file('cand_hyps.18771.bin')
    # hyp_tree = cand_list[3].tree
    #
    # ast_tree = decode_tree_to_ast(hyp_tree)
    # print astor.to_source(ast_tree)

    pass
