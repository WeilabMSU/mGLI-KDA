# mGLI-KDA

<div align='center'>
 
<!-- [![preprint](https://img.shields.io/static/v1?label=arXiv&message=2310.12508&color=B31B1B)](https://www.google.com/) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**Title** - Knot data analysis using multiscale Gauss link integral.

**Authors** - Li Shen, Hongsong Feng, Fengling Li, Fengchun Lei, Jie Wu, and Guo-Wei Wei

---

## Table of Contents

- [TopoFormer](#topoformer)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Model Architecture](#model-architecture)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Datasets](#datasets)
  - [Preparing Topologial Sequence](#preparing-topologial-sequence)
  - [Fine-Tuning Procedure for Customized Data](#fine-tuning-procedure-for-customized-data)
  - [Results](#results)
      - [Pretrained models](#pretrained-models)
      - [Finetuned models and performances](#finetuned-models-and-performances)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

---

## Introduction

In the past decade, topological data analysis (TDA) has emerged as a powerful algebraic topology approach in data science. The main technique used in TDA is persistent homology, which tracks topological invariants over the filtration of point cloud data. Although knot theory and related subjects are a focus of study in mathematics, their success in practical applications is quite limited due to the lack of localization and quantization. We address these challenges by introducing knot data analysis (KDA), a new paradigm that incorporates curve segmentation and multiscale analysis into the Gauss link integral.  The resulting multiscale Gauss link integral (mGLI) recovers the global topological properties of knots and links at an appropriate scale and offers a multiscale geometric topology approach to capture the local structures and connectivities in data. The proposed mGLI significantly outperforms other state-of-the-art methods across various benchmark problems in 14  intricately complex biological datasets, including protein flexibility analysis, protein-ligand interactions, hERG potassium channel blockade screening, and quantitative toxicity assessment. Our KDA opens a new research area in data science.

> **Keywords**: Knot data analysis, Gauss link integral, multiscale analysis.

---

## Model Architecture

Schematic illustration of the overall mGLI-based knot data analysis (KDA) platform is shown in below.

![Model Architecture](concepts.pdf)

Further explain the details in the [paper](https://github.com/WeilabMSU/mGLI-KDA), providing context and additional information about the architecture and its components.

---

## Getting Started

### Prerequisites

- numpy                     1.21.0
- scipy                     1.7.3
- pytorch                   1.10.0 
- pytorch-cuda              11.7
- torchvision               0.11.1
- scikit-learn              1.0.2
- python                    3.10.12

### Installation

```
git clone https://github.com/WeilabMSU/TopoFormer.git
```

---

## Datasets

A brief introduction about the benchmarks.

| Datasets                |Total    | Training Set                 | Test Set                                             |
|-|-----------------------------|------------------------------|------------------------------                        |
| Set-364 | 364       |   -    |      -                                                            |
| B-factor (small) | 30       |   -    |      -                                                            |
| B-factor (medium) | 36       |   -    |      -                                                            |
| B-factor (large) | 34       |   -    |      -                                                            |
| PDBbind-v2007       |1300 |1105  [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                        | 195 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                         |
| PDBbind-v2013       |2959|2764  [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                        | 195 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                         |
| PDBbind-v2016       |4057|3767  [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                        | 290 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                         |
| IGC50       |1792|1434  [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                        | 358 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                         |
| LC50       |823|659  [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                        | 164 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                         |
| LC50DM       |353|283  [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                        | 70 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                         |
| LD50       |7413|5931  [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                        | 1482 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                         |
| Zhang data       |1334|927  [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                        | 407 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                         |
| Li data       |4813|3721  [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                        | 1092 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                         |
| Cai data       |4447|3954  [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                        | 493 [Label](https://weilab.math.msu.edu/Downloads/TopoFormer/Benchmarks_labels.zip)                         |



- RowData: the protein-ligand complex structures. From PDBbind
- TopoFeature: the topological embedded features for the protein-ligand complex. All features are saved in a dict, which `key` is the protein ID, and `value` is the topological embedded features for corresponding complex. The downloaded file is .zip file, which contains two file (1) `TopoFeature_large.npy`: topological embedded features with a filtration parameter ranging from 0 to 10 and incremented in steps of 0.1 \AA; (2) `TopoFeature_small.npy`: topological embedded features with a filtration parameter ranging from 2 to 12 and incremented in steps of 0.2 \AA; 
- Label: the .csv file, which contains the protein ID and corresponding binding affinity.

---
## Preparing Topologial Sequence

```shell
# get the usage
python ./code_pkg/main_potein_ligand_topo_embedding.py -h

# examples
python ./code_pkg/main_potein_ligand_topo_embedding.py --output_feature_folder "../examples/output_topo_seq_feature_result" --protein_file "../examples/protein_ligand_complex/1a1e/1a1e_pocket.pdb" --ligand_file "../examples/protein_ligand_complex/1a1e/1a1e_ligand.mol2" --dis_start 0 --dis_cutoff 5 --consider_field 20
```


## Fine-Tuning Procedure for Customized Data

```shell
bs=32 # batch size
lr=0.00008  # learning rate
ms=10000  # max training steps
fintuning_python_script=./code_pkg/topt_regression_finetuning.py
model_output_dir=./outmodel_finetune_for_regression
mkdir $model_output_dir
pretrained_model_dir=./pretrained_model
scaler_path=./code_pkg/pretrain_data_standard_minmax_6channel_large.sav
validation_data_path=./CASF_2016_valid_feat.npy
train_data_path=./CASF_2016_train_feat.npy
validation_label_path=./CASF2016_core_test_label.csv
train_label_path=./CASF2016_refine_train_label.csv

# finetune for regression on one GPU
CUDA_VISIBLE_DEVICES=1 python $fintuning_python_script --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 --num_train_epochs 100 --max_steps $ms --per_device_train_batch_size $bs --base_learning_rate $lr --output_dir $model_output_dir --model_name_or_path $pretrained_model_dir --scaler_path $scaler_path --validation_data $validation_data_path --train_data $train_data_path --validation_label $validation_label_path --train_label $train_label_path --pooler_type cls_token --random_seed 1234 --seed 1234;
```


```shell
# script for no validation data and validation label
# docking and screening
bs=32 # batch size
lr=0.0001  # learning rate
ms=5000  # max training steps
fintuning_python_script=./code_pkg/topt_regression_finetuning_docking.py
model_output_dir=./outmodel_finetune_for_docking
mkdir $model_output_dir
pretrained_model_dir=./pretrained_model
scaler_path=./code_pkg/pretrain_data_standard_minmax_6channel_filtration50-12.sav
train_data_path=./train_feat.npy
train_label_path=./train_label.csv

# finetune for regression on one GPU
CUDA_VISIBLE_DEVICES=1 python $fintuning_python_script --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 --num_train_epochs 100 --max_steps $ms --per_device_train_batch_size $bs --base_learning_rate $lr --output_dir $model_output_dir --model_name_or_path $pretrained_model_dir --scaler_path /$scaler_path --train_data $train_data_path --train_label $train_label_path --validation_data None --validation_label None --train_val_split 0.1 --pooler_type cls_token --random_seed 1234 --seed 1234 --specify_loss_fct 'huber';
```


---

## Results

#### Pretrained models
- Pretrained TopoFormer model large. [Download](https://weilab.math.msu.edu/Downloads/TopoFormer/TopoFormer_s_pretrained_model.zip)
- Pretrained TopoFormer model small. [Download](https://weilab.math.msu.edu/Downloads/TopoFormer/TopoFormer_pretrained_model.zip)

#### Finetuned models and performances
- Scoring


| Finetuned for scoring                                                | Training Set                  | Test Set| PCC | RMSE (kcal/mol) |
|-------------------------------------------------                     |-------------                  |---------|-    |-                |
| CASF-2007 [result](./Results)      | 1105                          | 195     |0.837|1.807|
| CASF-2007 small [result](./Results)| 1105                          | 195     |0.839|1.807|
| CASF-2013 [result](./Results)      | 2764                          | 195     |0.816|1.859|
| CASF-2016 [result](./Results)      | 3772                          | 285     |0.864|1.568|
| PDB v2016 [result](./Results)      | 3767                          | 290     |0.866|1.561|
| PDB v2020 [result](./Results)      | 18904 <br> (exclude core sets)|195<br>CASF-2007 core set|0.853|1.295|
|                                    |                               |195<br>CASF-2013 core set|0.832|1.301|
|                                    |                               |285<br>CASF-2016 core set|0.881|1.095|

Note, there are 20 TopoFormers are trained for each dataset with distinct random seeds to address initialization-related errors. And 20 gradient boosting regressor tree (GBRT) models are subsequently trained one these sequence-based features, which predictions can be found in the [results](./Results) folder. Then, 10 models were randomly selected from TopoFormer and GBDT models, respectively, the consensus predictions of these models was used as the final prediction result. The performance shown in the table is the average result of this process performed 400 times.

- Docking


| Finetuned for docking                                                | Success rate |
|-------------------------------------------------                     |-             |
| CASF-2007 [result](./Results)| 93.3%         |
| CASF-2013 [result](./Results)| 91.3%         |

- Screening

| Finetuned for screening                                              |Success rate on 1%|Success rate on 5%|Success rate on 10%|EF on 1%|EF on 5%|EF on 10%|
|-                                                                     | - | - | - | - | - | - |
| CASF-2013 |68%|81.5%|87.8%|29.6|9.7|5.6|

Note, the EF here means the enhancement factor. Each target protein has a finetuned model. [result](./Results) contains all predictions.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code or the pre-trained models in your work, please cite our work. 
- Chen, Dong, Jian Liu, and Guo-Wei Wei. "TopoFormer: Multiscale Topology-enabled Structure-to-Sequence Transformer for Protein-Ligand Interaction Predictions."

---

## Acknowledgements

This project has benefited from the use of the [Transformers](https://github.com/huggingface/transformers) library. Portions of the code in this project have been modified from the original code found in the Transformers repository.
