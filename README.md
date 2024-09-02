# Bucketed Ranking-based Losses for Efficient Training of Object Detectors

The official implementation of Bucketed Ranking-based Losses. Our implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

> [**Bucketed Ranking-based Losses for Efficient Training of Object Detectors**](https://arxiv.org/abs/2407.14204),            
> Feyza Yavuz, Baris Can Cam, Adnan Harun Dogan, [Kemal Oksuz](https://kemaloksuz.github.io/), [Emre Akbas](http://user.ceng.metu.edu.tr/~emre/), [Sinan Kalkan](http://www.kovan.ceng.metu.edu.tr/~sinan/),
> *ECCV 2024. ([arXiv pre-print](https://arxiv.org/abs/2407.14204))*



## Introduction

**What is Bucketed Ranking-based (BR) Losses?** Bucketing for ranking-based losses enhances the efficiency of such losses in object detection by grouping negative predictions into buckets, significantly reducing the number of pairwise comparisons required during training. Bucketing maintains the alignment with evaluation criteria and robustness against class imbalance of ranking-based loss functions while drastically improving the time complexity.

<p align="center">
  <img src="figures/ranking_comparison_2.png" width="600">
</p>

**Benefits of BR Loss on Efficiency and Simplification of Training.** With BR Loss, we achieve significant improvements in training efficiency: (i) The bucketed approach reduces the time complexity to O(max(N log(N),P²)), allowing faster training, (ii) BR Loss maintains the simplicity and robustness of ranking-based approaches without requiring complex sampling heuristics or additional auxiliary heads, and (iii) it enables efficient training of large-scale object detectors, including transformer-based models, with minimal tuning.

**Benefits of BR Loss on Improving Performance.** Using BR Loss, we train seven diverse visual detectors and demonstrate consistent performance improvements: (i) BR Loss accelerates training by 2× on average while preserving the accuracy of unbucketed versions, (ii) For the first time, we successfully train transformer-based detectors like CoDETR using ranking-based losses, consistently outperforming their original configurations across multiple backbones.

<p align="center">
  <img src="figures/performance_comparison.png" width="600">
</p>

## Model Zoo

## Running

### Install
We implement BucketedRankingBasedLosses using [MMDetection V2.25.3](https://github.com/open-mmlab/mmdetection/releases/tag/v2.25.3) and [MMCV V1.5.0](https://github.com/open-mmlab/mmcv/releases/tag/v1.5.0).
The source code of MMdetection has been included in this repo and you only need to build MMCV following [official instructions](https://github.com/open-mmlab/mmcv/tree/v1.5.0#installation).
We test our models under ```python=3.7.11,pytorch=1.11.0,cuda=11.3```. 

### Data
The COCO dataset and LVIS dataset should be organized as:
```
BucketedRankingBasedLosses
└── data
    ├── coco
    │   ├── annotations
    │   │      ├── instances_train2017.json
    │   │      └── instances_val2017.json
    │   ├── train2017
    │   └── val2017
    │
    └── lvis_v1
        ├── annotations
        │      ├── lvis_v1_train.json
        │      └── lvis_v1_val.json
        ├── train2017
        └── val2017        
```

### Training


### Testing


## Cite 

If you find this repository useful, please use the following BibTeX entry for citation.

```latex

```

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
