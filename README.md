# CAT-probing: A Metric-based Approach to Interpret How Pre-trained Models for Programming Language Attend Code Structure

## Updates

- TBD: our code :bookmark_tabs:
- 2022/10/06: Release the paper of CAT-probing, check out our [paper](https://arxiv.org/abs/xxxx.yyyy). :clap:
- 2022/10/06: CAT-probing is accepted by **Findings of EMNLP 2022** :tada:



## Introduction

TBD



More details are provided in our EMNLP22 paper and our arXiv paper [CAT-probing: A Metric-based Approach to Interpret How Pre-trained Models for Programming Language Attend Code Structure](CAT-probing: A Metric-based Approach to Interpret How Pre-trained Models for Programming Language Attend Code Structure).

## Environment



```bash
conda create --name cat python=3.8
conda activate cat
pip install transformers==4.1.1
pip install datasets
pip install sklearn
git clone https://github.com/QiushiSun/CAT-Probing
cd CAT-Probing
```



## Using CAT-probing

### Preparing Data

The dataset we use comes from [CodeSearchNet](https://arxiv.org/pdf/1909.09436.pdf) and the dataset is filtered as the following:

- Remove examples that codes cannot be parsed into an abstract syntax tree.
- Remove examples that #tokens of documents is < 3 or >256
- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
- Remove examples that documents are not English.

```bash
unzip dataset.zip
mkdir data
cd data
mkdir summarize
cd summarize
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ../..
```

### Train

```
export MODEL_NAME=
export TASK="summarize"
export SUB_TASK=
bash run.sh $MODEL_NAME $TASK $SUB_TASK
```

`MODEL_NAME` can be any one of `["roberta", "codebert", "graphcodebert", "unixcoder"]`.

`SUB_TASK` can be any one of `["go", "java", "javascript", "python"]`.



### Probing





### Visualization





## Citation

If you use this code or UniXcoder, please consider citing us.:point_down:

```
@article{TBD,
  title={CAT-probing: A Metric-based Approach to Interpret How Pre-trained Models for Programming Language Attend Code Structure},
  author={Chen, Nuo and Sun, Qiushi and Zhu, Renyu and Xiang, Li and Xuesong, Lu and Ming, Gao},
  journal={arXiv preprint arXiv:TBD},
  year={2022}
}
```



