import os
import sys
import json
from random import random, randint


def get_swap_dict(formula_list):
    swap_dict = {}
    for _, formula in formula_list.items():
        for (nt1, nv1, _), (nt2, nv2, _), (_, ) in formula:
            if nv1 != 0:
                if nt1 not in swap_dict:
                    swap_dict[nt1] = set()
                swap_dict[nt1].add(nv1)
            if nv2 != 0:
                if nt2 not in swap_dict:
                    swap_dict[nt2] = set()
                swap_dict[nt2].add(nv2)
    for key in swap_dict:
        swap_dict[key] = list(swap_dict[key])
    return swap_dict


def swap_value(formula_list, p = 0.3):
    swap_dict = get_swap_dict(formula_list)
    new_formula_list = {}
    for name, formula in formula_list.items():
        new_formula = []
        node_dict = {}
        for (nt1, nv1, id1), (nt2, nv2, id2), (edge, ) in formula:
            if id1 not in node_dict.keys():
                if nt1 in swap_dict and random() < p:
                    nv1 = swap_dict[nt1][randint(0, len(swap_dict[nt1]) - 1)]
                node1 = (nt1, nv1, id1)
                node_dict[id1] = node1
            else:
                node1 = node_dict[id1]
            
            if id2 not in node_dict.keys():
                if nt2 in swap_dict and random() < p:
                    nv2 = swap_dict[nt2][randint(0, len(swap_dict[nt2]) - 1)]
                node2 = (nt2, nv2, id2)
                node_dict[id2] = node2
            else:
                node2 = node_dict[id2]

            new_formula.append((node1, node2, (edge, )))
        new_formula_list[name] = new_formula
    return new_formula_list


def main(dataset_folder):
    with open(os.path.join(dataset_folder, 'opt_list.txt')) as f:
        opt_data = json.loads(f.read())
    with open(os.path.join(dataset_folder, 'slt_list.txt')) as f:
        slt_data = json.loads(f.read())

    opt_swap_data = get_swap_dict(opt_data)
    slt_swap_data = get_swap_dict(slt_data)

    with open(os.path.join(dataset_folder, 'opt_swap_dict.txt'), 'w') as f:
        f.write(json.dumps(opt_swap_data))
    with open(os.path.join(dataset_folder, 'slt_swap_dict.txt'), 'w') as f:
        f.write(json.dumps(slt_swap_data))


if __name__ == '__main__':
    _, dataset_folder, = sys.argv
    main(dataset_folder)
        

    
    
