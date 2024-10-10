# GCL-Formula-Retrieval-Variable-Sub

## Setup
### Packages
Tested under Python 3.9.7 (torch 2.1.0 / cuda 12.1) on Windows.
Evaluation is performed on Ubuntu 22.04. (Due to building issues on Windows.)
Required packages are listed in `requirements.txt`

### Evaluation tool
Evaluation is performed using [trec_eval](https://github.com/usnistgov/trec_eval). 
The trec_eval executable should be in the same folder as `results.bash`.

## File Description
Download data: Please see [ecir-2020](https://drive.google.com/drive/folders/1emboT7k4m7yKjru3AOb1xScZgbUnQuC8).

### `datasets`
Retrieval result files are saved under the following directory structure:
```
datasets/
|── NTCIR12_latex_expressions.txt
|── NTCIR12_MathWiki-qrels_judge.dat
|── topics.txt
└── encoder/
```

Dataset folders (like `.\datasets\encoder\`) should be placed under `.\datasets\`.
In said folder, the following 8 files should be presented:
1. opt_char_embeding.txt: Feature embeddings using [Tangent-CFT](https://github.com/BehroozMansouri/TangentCFT) in OPT form
2. slt_char_embeding.txt: Feature embeddings using [Tangent-CFT](https://github.com/BehroozMansouri/TangentCFT) in SLT form
3. opt_list.txt: Formula paths in OPT form
4. slt_list.txt: Formula paths in SLT form
5. query_opt_list.txt: Query formula paths in OPT form
6. query_slt_list.txt: Query formula paths in SLT form
7. opt_judge: Judged formula paths in OPT form (formulas presented in `NTCIR12_MathWiki-qrels_judge.dat`)
8. slt_judge: Judged formula paths in SLT form (formulas presented in `NTCIR12_MathWiki-qrels_judge.dat`)

- NTCIR12_latex_expressions.txt: Latex expressions of the formulas.
- NTCIR12_MathWiki-qrels_judge.dat: Manually evaluated formulas for judging.
- topics.txt: Query formulas.


## Quick Start
### Preparing the dataset
Navigate to the dataset directory and unzip the following files:
```
$ cd datasets/encoder
$ tar zxvf opt_list.txt.tgz
$ tar zxvf slt_list.txt.tgz
```

### Training
Train with `train.ps1`：
* Usage: 
    ```
    > .\train.ps1 <dataset_dir> <result_ouput_dir>
    ```
* Example: 
    ```
    > .\train.ps1 ".\dataset\encoder\" ".\Retrieval_result\"
    ```

### Evaluation
> Evaluation is done on Ubuntu.
#### Retrieval Result Files Structure
Retrieval result files are saved under the following directory structure:
```
<root>/
└── <graph_encode_form>/
    └── <aug_id>/
        └── <batch size>/
            └── <run id>/
                |── retrieval_res_<encode>_<aug_id>_<batch_size>
                └── model
```

#### Evaluating
> [This](http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system) should explain almost everything.
> `trec_eval` should be placed under the same directory as `result.bash`.

- Using `trec_eval` directly
    - for full relevant bpref results, add `-l3` flag
    - Usage:
        ```
        $ ./trec_eval -m <bpref/ndcg> <qrel file> <retrieval file path>
        ```
- Using `result.bash`
    - Going through all result files under a directory
    - Usage:
        ```
        $ ./results.bash <qrel file> <result_dir>
        ```


