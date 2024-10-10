import os
import torch
import os.path as osp
import torch.nn.functional as F
import numpy as np

class FormulaRetrieval(): 
    def __init__(self, series, query_emb, top_k = 1000):
        super(FormulaRetrieval, self).__init__()
        self.retrieval_result = {}
        self.emb_dict_query = query_emb
        self.emb_dict_series = series
        self.retrieval(self.emb_dict_query, self.emb_dict_series, top_k)
        
    def batch_detatch(self, batch_index, embs, emb_dict):
        tmp_dict = dict(zip(batch_index, embs))
        emb_dict.update(tmp_dict)
        return emb_dict
   
    def retrieval(self, emb_dict_query, emb_dict_series, k):
        print("retrieval...")
        formula_index = list(emb_dict_series.keys())
        series_tensor = torch.as_tensor(np.array(list(emb_dict_series.values())))        
        for query_key in emb_dict_query:
            query = emb_dict_query[query_key]
            query_tensor = torch.tensor(query).double()
            self.retrieval_result[query_key] = self.get_formula_retrieval(series_tensor, formula_index, query_tensor, k)

    def get_formula_retrieval(self, series_tensor, formula_index, query_tensor, k):
        ## dim problem, now fixed
        comp = query_tensor.tile(series_tensor.size(0)).reshape((-1, query_tensor.size(0)))
        dist = F.cosine_similarity(series_tensor, comp)
        index_sorted = torch.sort(dist, descending=True)[1]
        top_k = index_sorted[:k]
        top_k = top_k.data.cpu().numpy()
        cos_values = torch.sort(dist, descending=True)[0][:k].data.cpu().numpy()
        result = {}
        count = 0
        for x in top_k:
            doc_id = formula_index[x]
            score = cos_values[count]
            result[doc_id] = score
            count += 1
        return result

    def create_retrieval_file(self, result_root, encode, aug_id, batch_size, run_id):
        print("create result file...")
        file_dir = os.path.join(result_root, f'{encode}/{aug_id}/{batch_size}/{run_id}')
        filename = f"/retrieval_res_{encode}_{aug_id}_{batch_size}"
        
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with open(file_dir + filename, "w", encoding = "utf-8") as file:
            for query in self.retrieval_result:
                count = 1
                line = query + " xxx "
                for s in self.retrieval_result[query]:
                    score = self.retrieval_result[query][s]
                    temp = line + s  + " " + str(count) + " " + str(score) + " Run_" + str(run_id)
                    count += 1
                    file.write(temp + "\n")
    
                