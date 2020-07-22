# Paraphrasing Complex Network: Network Compression via Factor Transfer (NeurIPS 2018)

This repository is the official implementation of [Paraphrasing Complex Network: Network Compression via Factor Transfer (FT)](https://arxiv.org/abs/1802.04977). 
The source code is for the experiment of ResNet on CIFAR-10. In this experiment, we use ResNet56 as a teacher network and ResNet20 as a student network on CIFAR-10 Dataset.

Before training the student network, pre-trained teacher network and paraphraser are needed. More details are in the paper.
We published FT in NeurIPS 2018. See our paper [here](https://papers.nips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer).
## Requirements

To install requirements using [environment.yml](environment.yml) refer to the [documentation.](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)


## Training

[train_student.py](train_student.py) is the code for training student with pretrained teacher and paraphraser using **Factor Transfer (FT)**. 
To train the model(s), run this command:

``` 
python train_student.py  --cu_num 0 
```

## Citation
Please refer to the following citation if this work is useful for your research.

### Bibtex:

```
@inproceedings{kim2018paraphrasing,
  title={Paraphrasing complex network: network compression via factor transfer},
  author={Kim, Jangho and Park, SeongUk and Kwak, Nojun},
  booktitle={Proceedings of the 32nd International Conference on Neural Information Processing Systems},
  pages={2765--2774},
  year={2018}
}
```

