# Baseline Results (AOT / AOST)

This file contains the baseline results for Video Object Segmentation (VOS) on the DAVIS and YouTube-VOS datasets, as reported in the AOT/AOST paper (arXiv:2203.11442) and its official repository.

## YouTube-VOS 2018 Validation

| Model | Stage | FPS | All-F | Mean | J Seen | F Seen | J Unseen | F Unseen |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AOTT | PRE_YTB_DAV | 41.0 | | 80.2 | 80.4 | 85.0 | 73.6 | 81.7 |
| AOTT | PRE_YTB_DAV | 41.0 | √ | 80.9 | 80.0 | 84.7 | 75.2 | 83.5 |
| DeAOTT | PRE_YTB_DAV | 53.4 | | 82.0 | 81.6 | 86.3 | 75.8 | 84.2 |
| AOTS | PRE_YTB_DAV | 27.1 | | 82.9 | 82.3 | 87.0 | 77.1 | 85.1 |
| AOTS | PRE_YTB_DAV | 27.1 | √ | 83.0 | 82.2 | 87.0 | 77.3 | 85.7 |
| DeAOTS | PRE_YTB_DAV | 38.7 | | 84.0 | 83.3 | 88.3 | 77.9 | 86.6 |
| AOTB | PRE_YTB_DAV | 20.5 | | 84.0 | 83.2 | 88.1 | 78.0 | 86.5 |
| AOTB | PRE_YTB_DAV | 20.5 | √ | 84.1 | 83.6 | 88.5 | 78.0 | 86.5 |
| DeAOTB | PRE_YTB_DAV | 30.4 | | 84.6 | 83.9 | 88.9 | 78.5 | 87.0 |
| AOTL | PRE_YTB_DAV | 16.0 | | 84.1 | 83.2 | 88.2 | 78.2 | 86.8 |
| AOTL | PRE_YTB_DAV | 6.5 | √ | 84.5 | 83.7 | 88.8 | 78.4 | 87.1 |
| DeAOTL | PRE_YTB_DAV | 24.7 | | 84.8 | 84.2 | 89.4 | 78.6 | 87.0 |
| R50-AOTL | PRE_YTB_DAV | 14.9 | | 84.6 | 83.7 | 88.5 | 78.8 | 87.3 |
| R50-AOTL | PRE_YTB_DAV | 6.4 | √ | 85.5 | 84.5 | 89.5 | 79.6 | 88.2 |
| R50-DeAOTL | PRE_YTB_DAV | 22.4 | | 86.0 | 84.9 | 89.9 | 80.4 | 88.7 |
| SwinB-AOTL | PRE_YTB_DAV | 9.3 | | 84.7 | 84.5 | 89.5 | 78.1 | 86.7 |
| SwinB-AOTL | PRE_YTB_DAV | 5.2 | √ | 85.1 | 85.1 | 90.1 | 78.4 | 86.9 |
| SwinB-DeAOTL | PRE_YTB_DAV | 11.9 | | 86.2 | 85.6 | 90.6 | 80.0 | 88.4 |

## YouTube-VOS 2019 Validation

| Model | Stage | FPS | All-F | Mean | J Seen | F Seen | J Unseen | F Unseen |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AOTT | PRE_YTB_DAV | 41.0 | | 80.0 | 79.8 | 84.2 | 74.1 | 82.1 |
| AOTT | PRE_YTB_DAV | 41.0 | √ | 80.9 | 79.9 | 84.4 | 75.6 | 83.8 |
| DeAOTT | PRE_YTB_DAV | 53.4 | | 82.0 | 81.2 | 85.6 | 76.4 | 84.7 |
| AOTS | PRE_YTB_DAV | 27.1 | | 82.7 | 81.9 | 86.5 | 77.3 | 85.2 |
| AOTS | PRE_YTB_DAV | 27.1 | √ | 82.8 | 81.9 | 86.5 | 77.3 | 85.6 |
| DeAOTS | PRE_YTB_DAV | 38.7 | | 83.8 | 82.8 | 87.5 | 78.1 | 86.8 |
| AOTB | PRE_YTB_DAV | 20.5 | | 84.0 | 83.1 | 87.7 | 78.5 | 86.8 |
| AOTB | PRE_YTB_DAV | 20.5 | √ | 84.1 | 83.3 | 88.0 | 78.2 | 86.7 |
| DeAOTB | PRE_YTB_DAV | 30.4 | | 84.6 | 83.5 | 88.3 | 79.1 | 87.5 |
| AOTL | PRE_YTB_DAV | 16.0 | | 84.0 | 82.8 | 87.6 | 78.6 | 87.1 |
| AOTL | PRE_YTB_DAV | 6.5 | √ | 84.2 | 83.0 | 87.8 | 78.7 | 87.3 |
| DeAOTL | PRE_YTB_DAV | 24.7 | | 84.7 | 83.8 | 88.8 | 79.0 | 87.2 |
| R50-AOTL | PRE_YTB_DAV | 14.9 | | 84.4 | 83.4 | 88.1 | 78.7 | 87.2 |
| R50-AOTL | PRE_YTB_DAV | 6.4 | √ | 85.3 | 83.9 | 88.8 | 79.9 | 88.5 |
| R50-DeAOTL | PRE_YTB_DAV | 22.4 | | 85.9 | 84.6 | 89.4 | 80.8 | 88.9 |
| SwinB-AOTL | PRE_YTB_DAV | 9.3 | | 84.7 | 84.0 | 88.8 | 78.7 | 87.1 |
| SwinB-AOTL | PRE_YTB_DAV | 5.2 | √ | 85.3 | 84.6 | 89.5 | 79.3 | 87.7 |
| SwinB-DeAOTL | PRE_YTB_DAV | 11.9 | | 86.1 | 85.3 | 90.2 | 80.4 | 88.6 |

## DAVIS 2017 Validation

| Model | Stage | FPS | Mean | J Score | F Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| AOTT | PRE_YTB_DAV | 51.4 | 79.2 | 76.5 | 81.9 |
| AOTS | PRE_YTB_DAV | 40.0 | 82.1 | 79.3 | 84.8 |
| AOTB | PRE_YTB_DAV | 29.6 | 83.3 | 80.6 | 85.9 |
| AOTL | PRE_YTB_DAV | 18.7 | 83.6 | 80.8 | 86.3 |
| R50-AOTL | PRE_YTB_DAV | 18.0 | 85.2 | 82.5 | 87.9 |
| SwinB-AOTL | PRE_YTB_DAV | 12.1 | 85.9 | 82.9 | 88.9 |

## DAVIS 2017 Test-dev

| Model | Stage | FPS | Mean | J Score | F Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| AOTT | PRE_YTB_DAV | 51.4 | 73.7 | 70.0 | 77.3 |
| AOTS | PRE_YTB_DAV | 40.0 | 75.2 | 71.4 | 78.9 |
| AOTB | PRE_YTB_DAV | 29.6 | 77.4 | 73.7 | 81.1 |
| AOTL | PRE_YTB_DAV | 18.7 | 79.3 | 75.5 | 83.2 |
| R50-AOTL | PRE_YTB_DAV | 18.0 | 79.5 | 76.0 | 83.0 |
| SwinB-AOTL | PRE_YTB_DAV | 12.1 | 82.1 | 78.2 | 85.9 |

## DAVIS 2016 Validation

| Model | Stage | FPS | Mean | J Score | F Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| AOTT | PRE_YTB_DAV | 51.4 | 87.5 | 86.5 | 88.4 |
| AOTS | PRE_YTB_DAV | 40.0 | 89.6 | 88.6 | 90.5 |
| AOTB | PRE_YTB_DAV | 29.6 | 90.9 | 89.6 | 92.1 |
| AOTL | PRE_YTB_DAV | 18.7 | 91.1 | 89.5 | 92.7 |
| R50-AOTL | PRE_YTB_DAV | 18.0 | 91.7 | 90.4 | 93.0 |
| SwinB-AOTL | PRE_YTB_DAV | 12.1 | 92.2 | 90.6 | 93.8 |
