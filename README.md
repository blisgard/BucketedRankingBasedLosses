# Bucketed Ranking-based Losses for Efficient Training of Object Detectors

[![arXiv](https://img.shields.io/badge/arXiv-2405.20459-b31b1b.svg)](https://arxiv.org/abs/2407.14204)

The official implementation of Bucketed Ranking-based Losses. Our implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

> [**Bucketed Ranking-based Losses for Efficient Training of Object Detectors**](https://arxiv.org/abs/2407.14204),            
> Feyza Yavuz, Baris Can Cam, Adnan Harun Dogan, [Kemal Oksuz](https://kemaloksuz.github.io/), [Emre Akbas](http://user.ceng.metu.edu.tr/~emre/), [Sinan Kalkan](http://www.kovan.ceng.metu.edu.tr/~sinan/),
> *ECCV 2024. ([arXiv pre-print](https://arxiv.org/abs/2407.14204))*



## Introduction

**What is Bucketed Ranking-based (BR) Losses?** Bucketing for ranking-based losses enhances the efficiency of such losses in object detection by grouping negative predictions into buckets, significantly reducing the number of pairwise comparisons required during training. Bucketing maintains the alignment with evaluation criteria and robustness against class imbalance of ranking-based loss functions while drastically improving the time complexity.

<p align="center">
  <img src="figures/ranking_comparison_2.png" width="600">
</p>

**BRS-DETR: Efficient and Robust Transformer-Based Object Detection with Bucketed Ranking-Based Losses** BRS-DETR integrates Bucketed Ranking-Based Loss (BRS Loss) into Co-DETR, delivering superior performance and training efficiency on the COCO benchmark. (i) BRS-DETR achieves a 0.8 AP improvement on ResNet-50 and consistent gains across other transformer-based backbones. (ii) BRS-DETR provides faster training: cuts training time by 6×, optimizing the handling of positive examples and loss calculation of auxillary heads.

**Benefits of BR Loss on Efficiency and Simplification of Training.** With BR Loss, we achieve significant improvements in training efficiency: (i) The bucketed approach reduces the time complexity to O(max(N log(N),P²)), allowing faster training, (ii) BR Loss maintains the simplicity and robustness of ranking-based approaches without requiring complex sampling heuristics or additional auxiliary heads, and (iii) it enables efficient training of large-scale object detectors, including transformer-based models, with minimal tuning.

**Benefits of BR Loss on Improving Performance.** Using BR Loss, we train seven diverse visual detectors and demonstrate consistent performance improvements: (i) BR Loss accelerates training by 2× on average while preserving the accuracy of unbucketed versions, (ii) For the first time, we successfully train transformer-based detectors like CoDETR using ranking-based losses, consistently outperforming their original configurations across multiple backbones.

<p align="center">
  <img src="figures/performance_comparison.png" width="600">
</p>

## How to Cite

Please cite the paper if you benefit from our paper or the repository:
```
@inproceedings{BRLoss,
       title = {Bucketed Ranking-based Losses for Efficient Training of Object Detectors},
       author = {Feyza Yavuz and Baris Can Cam and Adnan Harun Dogan and Kemal Oksuz and Emre Akbas and Sinan Kalkan},
       booktitle = {European Conference on Computer Vision (ECCV)},
       year = {2024}
}
```
## Specifications of Dependencies and Preparation
- Please see [get_started.md](docs/en/get_started.md) for requirements and installation of mmdetection.
- Please see [introduction.md](docs/en/1_exist_data_model.md) for dataset preparation and basic usage of mmdetection.

Please note that, we implement our method on [MMDetection V2.25.3](https://github.com/open-mmlab/mmdetection/releases/tag/v2.25.3) and [MMCV V1.5.0](https://github.com/open-mmlab/mmcv/releases/tag/v1.5.0). More specifically, we use ```python=3.7.11, pytorch=1.11.0, cuda=11.3``` versions.

## Trained Models
Here, we report validation set results for object detection and instance segmentation tasks. For object detection we report results on COCO validation set. For instance segmentation we report results on both Cityscapes and LVIS validation sets.

We refer to the [RS Loss](http://github.com/kemaloksuz/RankSortLoss) repository for models trained with RS Loss. 

### Transformer-based Object Detection
#### Co-DETR
|    Backbone     |  Epoch |  Detector | box AP |  Log  | Config | Model |
| :-------------: | :-----: | :-----: | :------------: | :------------: | :-------: | :-------: |
|    ResNet-50 | 12 |  Co-DETR  | 49.3 |  [log](https://drive.google.com/file/d/19seqFA0zk8NB3x4CVyLArwRT4Ni7GYfh/view?usp=drive_link)| [config](projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py) | [model](https://drive.google.com/file/d/12lfjUyhivGFmyXgBRFqV8kgkbHWzkU03/view?usp=drive_link) |
|    ResNet-50 | 12 |  BRS-DETR  |  50.1 |  [log]()| [config]() | [model]() |
|    Swin-T | 12 |  Co-DETR  | 51.7 |  [log]()| [config](projects/configs/co_deformable_detr/co_deformable_detr_swin_tiny_1x_coco.py) | [model]() |
|    Swin-T | 12 |  BRS-DETR  |  52.3 |  [log]()| [config]() | [model]() |
|    Swin-L | 12 |  Co-DETR  | 56.9 |  [log]()| [config](projects/configs/co_deformable_detr/co_deformable_detr_swin_large_1x_coco.py) | [model]() |
|    Swin-L | 12 |  BRS-DETR  | 57.2 |  [log]()| [config]() | [model]() |

### Multi-stage Object Detection
#### Faster R-CNN

|    Backbone     |  Epoch |  Loss Func. | Time | box AP |  Log  | Config | Model |
| :-------------: | :-----: | :-----: | :------------: | :------------: | :------------: | :-------: | :-------: |
|    ResNet-50 | 12 |  RS  | 0.58 | 39.5 |  [log](https://drive.google.com/file/d/1Pq7Z5QMyl8kzM-0KylHCepDMG00lta3R/view?usp=drive_link)| [config](configs/ranksort_loss/ranksort_loss_faster_rcnn_r50_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1tnovgbk6FrHBNvveOzGmNzTSCyG7zuZR/view?usp=drive_link) |
|    ResNet-50 | 12 |  BRS  | 0.19 (3.0x &#8595;) | 39.5 |  [log](https://drive.google.com/file/d/1pgn-CooUhxE74KqohzxBQKEcXT_Avpbv/view?usp=drive_link)| [config](configs/bucketed_ranking_losses/bucketed_ranksort_loss_faster_rcnn_r50_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/10KgZb8UNrAjE66m67S9S8dELa1j6-xai/view?usp=drive_link) |
|    ResNet-101 | 36 |  RS  | 0.91 | 47.3 |  [log](https://drive.google.com/file/d/1kkunFtD6VciTkmWj1WAc0dXWJJGScqNb/view?usp=drive_link)| [config](configs/ranksort_loss/ranksort_loss_faster_rcnn_r101_fpn_mstrain_dcn_3x_coco.py) | [model](https://drive.google.com/file/d/12BLuFCdrpy7Y_fRUWcAIPu4Wx5OWtURk/view?usp=drive_link) |
|    ResNet-101 | 36 |  BRS  | 0.47 (2.0x &#8595;) | 47.7 |  [log]()| [config](configs/bucketed_ranking_losses/bucketed_ranksort_loss_faster_rcnn_r101_fpn_mstrain_dcn_3x_coco.py) | [model]() |

#### Cascade R-CNN

|    Backbone     |  Epoch |  Loss Func. | Time | box AP |  Log  | Config | Model |
| :-------------: | :-----: | :-----: | :------------: | :------------: | :------------: | :-------: | :-------: |
|    ResNet-50 | 12 |  RS  | 1.54 | 41.1 |  [log](https://drive.google.com/file/d/1vU_fqvm0IAJTEhoiH4CEjG3wMUXf-vE8/view?usp=drive_link)| [config](configs/ranksort_loss/ranksort_loss_cascade_rcnn_r50_fpn_1x_coco.py) | [model]() |
|    ResNet-50 | 12 |  BRS  | 0.29 (5.3x &#8595;) | 41.1 |  [log](https://drive.google.com/file/d/1B7MyNhVwcIid5HA7oBwhzexFIHMoTjnt/view?usp=drive_link)| [config](configs/bucketed_ranking_losses/bucketed_ranksort_loss_cascade_rcnn_r50_fpn_1x_coco.py) | [model]() |

### One-stage Object Detection

#### ATSS

|    Backbone     |  Epoch |  Loss Func. | Time | box AP |  Log  | Config | Model |
| :-------------: | :-----: | :-----: | :------------: | :------------: | :------------: | :-------: | :-------: |
|    ResNet-50 | 12 |  AP  | 0.34 | 38.3 |  [log](https://drive.google.com/file/d/1Q8aTpCcR3FDwQ3a8ngvZNgS0FWLlEd2o/view?usp=drive_link)| [config](configs/bucketed_ranking_losses/ap_loss_atss_r50_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1i_cSEfHa3gGE8hjwZ7YKBcUqXIulHD9R/view?usp=drive_link) |
|    ResNet-50 | 12 |  BAP  | 0.18 (1.9x &#8595;) | 38.5 |  [log](https://drive.google.com/file/d/1As3TOqCm0EC2Ub0mX4Buw7gGVvnqM5Nr/view?usp=drive_link)| [config](configs/bucketed_ranking_losses/bucketed_ap_loss_atss_r50_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1F-PzaAAEbd9qIem73IHDKC-ka7WNxmZo/view?usp=drive_link) |
|    ResNet-50 | 12 |  RS  | 0.44 | 39.8 |  [log](https://drive.google.com/file/d/1ohkUhOGcMzUB5P04yYjdMzK8iuL6MDOA/view?usp=drive_link)| [config](configs/ranksort_loss/ranksort_loss_atss_r50_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/12J36iTpGWLvttIsZc6BnlI6FwgimAFkd/view?usp=drive_link) |
|    ResNet-50 | 12 |  BRS  | 0.19 (2.4x &#8595;) | 39.8 |  [log](https://drive.google.com/file/d/1S2iV2SYGD1B8VxGZmrJX06EYWt55bxN5/view?usp=drive_link)| [config](configs/bucketed_ranking_losses/bucketed_ranksort_loss_atss_r50_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1bSd9CzhmnsSzpbzeHo7mFt8z4Y9audt3/view?usp=drive_link) |

#### PAA

|    Backbone     |  Epoch |  Loss Func. | Time | box AP |  Log  | Config | Model |
| :-------------: | :-----: | :-----: | :------------: | :------------: | :------------: | :-------: | :-------: |
|    ResNet-50 | 12 |  AP  | TODO | 37.3 |  [log]()| [config](configs/bucketed_ranking_losses/ap_loss_paa_r50_fpn_1x_coco.py) | [model]() |
|    ResNet-50 | 12 |  BAP  | TODO 1.5x &#8595; | 37.2 |  [log]()| [config](configs/bucketed_ranking_losses/bucketed_ap_loss_paa_r50_fpn_1x_coco.py) | [model]() |
|    ResNet-50 | 12 |  RS  | TODO | 40.8 |  [log]()| [config](configs/ranksort_loss/ranksort_loss_paa_r50_fpn_1x_coco.py) | [model]() |
|    ResNet-50 | 12 |  BRS  | TODO 1.9x &#8595; | 40.9 |  [log]()| [config](configs/bucketed_ranking_losses/bucketed_ranksort_loss_paa_r50_fpn_1x_coco.py) | [model]() |

### Instance Segmentation
We use Mask R-CNN as the baseline model to experiment with our method in the instance segmentation task.

#### Coco Val
|    Backbone     |  Epoch |  Loss Func. | Time | mask AP |  Log  | Config | Model |
| :-------------: | :-----: | :-----: | :------------: | :------------: | :------------: | :-------: | :-------: |
|    ResNet-50 | 12 |  RS  | 0.68 | 36.3 |  [log](https://drive.google.com/file/d/1Vzracn8lH1WY9ka_tWteaCrCyYD6h7so/view?usp=drive_link)| [config](configs/ranksort_loss/ranksort_loss_mask_rcnn_r50_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1YrKH_cJiL84TfQGU9Fgk0rrjxrjfD5qZ/view?usp=drive_link) |
|    ResNet-50 | 12 |  BRS  | 0.29 (2.3x &#8595;) | 36.2 |  [log](https://drive.google.com/file/d/1Jr2Nm-p5kiCavnLFM-qXouwrfki1wvh3/view?usp=drive_link)| [config](configs/bucketed_ranking_losses/bucketed_ranksort_loss_mask_rcnn_r50_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1dCJnhh61-7k8EemJu8rQ80XIGxRdzVVb/view?usp=drive_link) |
|    ResNet-101 | 36 |  RS  | 0.71 | 40.2 |  [log](https://drive.google.com/file/d/1f9sJIjugfzY05SrVagAL3d_vCJ_nlEJm/view?usp=drive_link)| [config](configs/ranksort_loss/ranksort_loss_mask_rcnn_r101_fpn_mstrain_3x_coco.py) | [model](https://drive.google.com/file/d/1-XCUUSjE3SgTd_gxOf6gcvpQP1WglHpG/view?usp=drive_link) |
|    ResNet-101 | 36 |  BRS  | 0.33 (2.2x &#8595;) | 40.3 |  [log](https://drive.google.com/file/d/15_P-IIQLSgSukDesT9D6uG0EIpLNaLF5/view?usp=drive_link)| [config](configs/bucketed_ranking_losses/bucketed_ranksort_mask_rcnn_r101_fpn_mstrain_3x_coco.py) | [model](https://drive.google.com/file/d/1CNcXbmjsdpH54DaAzy6jED74pdUaYbMV/view?usp=drive_link) |
#### Cityscapes
|    Backbone     |  Epoch |  Loss Func. | Time | box AP | mask AP | Log  | Config | Model |
| :-------------: | :-----: | :-----: | :------------: | :------------: |:------------: |:------------: | :-------: | :-------: |
|    ResNet-50 | 12 |  RS  | 0.43 | 43.7 | 38.2 |  [log](https://drive.google.com/file/d/1jTp2pJgDEee8YSrhkaAkDJbIo6YfAHxF/view?usp=drive_link)| [config](configs/ranksort_loss/ranksort_mask_rcnn_r50_fpn_1x_cityscapes.py) | [model](https://drive.google.com/file/d/1N3D4VGz4hgMlnj2YEYYeuTfm3xEeXJJG/view?usp=drive_link) |
|    ResNet-50 | 12 |  BRS  | 0.19 (2.3x &#8595;) | 43.3 | 38.5 | [log](https://drive.google.com/file/d/1X6734NkH8-IyVXipCjFBwwV3VaMc4p8o/view?usp=drive_link)| [config](configs/bucketed_ranking_losses/bucketed_ranksort_mask_rcnn_r50_fpn_1x_cityscapes.py) | [model](https://drive.google.com/file/d/102-bk0eHZKRwxa50vi-sB26bocJkINd3/view?usp=drive_link) |
#### LVIS
|    Backbone     |  Epoch |  Loss Func. | Time | mask AP |  Log  | Config | Model |
| :-------------: | :-----: | :-----: | :------------: | :------------: | :------------: | :-------: | :-------: |
|    ResNet-50 | 12 |  RS  | 0.87 | 25.6 |  [log](https://drive.google.com/file/d/1ygPh0F8PamRi_on8gOMCYGZwHG5eXKTc/view?usp=drive_link)| [config](configs/ranksort_loss/ranksort_mask_rcnn_r50_fpn_1x_lvis_v1.py) | [model](https://drive.google.com/file/d/1s1LBWfXPNjq3w0m5lM6NpXSisN6G0g11/view?usp=drive_link) |
|    ResNet-50 | 12 |  BRS  | 0.35 (2.5x &#8595;) | 25.8 |  [log](https://drive.google.com/file/d/1EQ5ZxNk1IEoBxVcQLyawz42HUxnCgqIV/view?usp=drive_link)| [config](configs/bucketed_ranking_losses/bucketed_ranksort_mask_rcnn_r50_fpn_1x_lvis_v1.py) | [model](https://drive.google.com/file/d/1aLNdDU5W5jkgu03Cs5NltDU1Y1FPTOjC/view?usp=drive_link) |

### License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
