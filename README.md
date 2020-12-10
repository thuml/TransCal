Code release for ["Transferable Calibration with Lower Bias and Variance in Domain Adaptation"](https://papers.nips.cc/paper/2020/hash/df12ecd077efc8c23881028604dbb8cc-Abstract.html)

## Dataset

### Office-31
Office-31 dataset can be found [here](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/). 

### Office-Home
Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

### DomainNet
DomainNet dataset can be found [here](https://ai.bu.edu/visda-2019/).

### VisDA-2017
VisDA-2017 dataset can be found [here](http://ai.bu.edu/visda-2017/)

### ImageNet-Sketch
ImageNet-Sketch dataset can be found [here](https://github.com/HaohanWang/ImageNet-Sketch)



## Calibration in DA:
#### Step 1: Train a domain adaptation model using the selected DA method
```
python 1_train_DA_models.py
```


#### Step 2: Fix the DA model and compute features for source train, source validation, and target, respectively

```
python 2_generate_features.py
```


#### Step 3: Transferable Calibration
```
python 3_TransCal.py
```
For a quick start, we provide the pre-trained features on Office-Home via CDAN+E here. You can directly skip the first two steps and run the third step to evaluate the performance of TransCal while calibrating this DA model on Office-Home.




## Citation
If you find this code useful for your research, please consider citing:
```
@inproceedings{Wang20TransCal,
    title = {Transferable Calibration with Lower Bias and Variance in Domain Adaptation},
    author = {Wang, Ximei and Long, Mingsheng and Wang, Jianmin and Jordan, Michael I},
    booktitle = {Advances in Neural Information Processing Systems 33},
    year = {2020}
}
```

## Contact
If you have any problem with our code, feel free to contact
- wxm17@mails.tsinghua.edu.cn
