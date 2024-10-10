import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from random import randint, random
from GCL.augmentors.augmentor import Graph, Augmentor

from time import time

class SwapNodeContent(Augmentor):
    def __init__(self, pe: float, char_emb: Dict[str, list], swap_dict: Dict[str, list], num_workers: int = 12):
        super(SwapNodeContent, self).__init__()
        self.pe = pe
        self.char_emb = {}
        self.emb_to_char = {}
        self.swap_dict = swap_dict
        self.num_workers = num_workers

        for k, v in char_emb.items():
            hv = tuple(v)
            self.char_emb[k] = hv
            self.emb_to_char[hv] = k

    def _task(self, n):
        res = n
        if random() < self.pe:
            n1, n2 = torch.split(n, n.size(0) // 2)
            char1 = self.emb_to_char[tuple(n1.tolist())]
            if e := self.swap_dict.get(char1):
                n2 = torch.tensor(self.char_emb[str(e[randint(0, len(e) - 1)])])
                res = torch.cat([n1, n2]) 
        return res

    def _augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        new_x = []
        for n in x:
            if random() < self.pe:
                n1, n2 = torch.split(n, n.size(0) // 2)
                char1 = self.emb_to_char[tuple(n1.cpu().tolist())]
                if e := self.swap_dict.get(char1):
                    n2 = torch.tensor(self.char_emb[str(e[randint(0, len(e) - 1)])]).to(n1.device)
                    new_x.append(torch.cat([n1, n2]))
                else:
                    new_x.append(n)
            else:
                new_x.append(n)
        
        return Graph(x=torch.stack(new_x), edge_index=edge_index, edge_weights=edge_weights)

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = x.cpu()
        
        with ThreadPoolExecutor(self.num_workers) as executor:
            new_x = list(executor.map(self._task, x))

        new_x = torch.stack(new_x).to(edge_index.device)
    
        return Graph(x=new_x, edge_index=edge_index, edge_weights=edge_weights)
