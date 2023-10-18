# CLIP-CARE

Here is more concise and neat version of CARE, and we only need to use CLIP for consistent sensing instead of training multiple diffrent models. The speed of the overall reasoning inference is almost the same with the case that only use the CLIP for main prediction.

## Introduction

Deep Neural Networks (DNNs) have revolutionized a multitude of machine learning applications but are notorious for their susceptibility to adversarial attacks. Our project introduces [**CARE (Certifiably Robust leArning with REasoning)**](https://arxiv.org/abs/2209.05055), aiming to enhance the robustness of DNNs by integrating them with reasoning abilities. This pipeline consists of two primary components:

- **Learning Component**: Consistently utilizes **CLIP** model for all different semantic predictions, e.g., recognizing if an input image contains something furry, instead of training multiple different models for sensing.

- **Reasoning Component**: Employs probabilistic graphical models like Markov Logic Networks (MLN) to apply domain-specific knowledge and logic reasoning to the learning process.

## Getting Started

### Prerequisites

`conda create -n name care python=3.9`

`conda activate care && pip install -r requirements.txt`

- install the pyg package following https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Citation & Further Reading

If you find our work useful, please consider citing our [paper](https://arxiv.org/abs/2209.05055):

```
@inproceedings{zhang2023care,
  title={CARE: Certifiably Robust Learning with Reasoning via Variational Inference},
  author={Zhang, Jiawei and Li, Linyi and Zhang, Ce and Li, Bo},
  booktitle={2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)},
  pages={554--574},
  year={2023},
  organization={IEEE}
}
```

## Contact

If you have any questions or encounter any errors while running the code, feel free to contact [jiaweiz@illinois.edu](mailto:jiaweiz@illinois.edu)!