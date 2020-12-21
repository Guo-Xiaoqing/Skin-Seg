# Complementary Network with Adaptive Receptive Fields for Melanoma Segmentation

by [Xiaoqing Guo](https://guo-xiaoqing.github.io/), [Zhen Chen](https://franciszchen.github.io/), [Yixuan Yuan](http://www.ee.cityu.edu.hk/~yxyuan/people/people.htm).

## Summary:
### Intoduction:
This repository is for our ISBI2020 paper ["Complementary Network with Adaptive Receptive Fields for Melanoma Segmentation"](https://arxiv.org/abs/2001.03893), which aims to solve the hole (Fig. 1 (a)) and shrink (Fig. 1 (b)) problem in predictions. The relatively low contrast between melanoma and non-melanoma regions confuses the network and causes the appearance of holes. The fuzzy boundaries lead to the shrinking prediction and further decrease the sensitivity of prediction. 


![](https://github.com/Guo-Xiaoqing/Skin-Seg/raw/master/intro_problem.png)
Fig. 1: Illustrations of (a) hole problem, (b) shrink problem. Each group includes the original image, ground truth and prediction of U-Net from left to right.

### Framework:
![](https://github.com/Guo-Xiaoqing/Skin-Seg/raw/master/framework.png)

## Usage:
### Requirement:
Tensorflow 1.4
Python 3.5

### Preprocessing:
Clone the repository:
```
git clone https://github.com/Guo-Xiaoqing/Skin-Seg.git
cd Skin-Seg
```

### Make tfrecord
* Please remember to augment dataset before make txt files

* Use "txt.py" to split training data and testing data. The generated txt files are showed in folder "./txt/".

* "make_tfrecords.py" is used to make tfrecord format data, which could be stored in folder "./skin2018/tfrecord/".

### Train the model: 
```
sh ./script/train_dml_mobilenet_on_market.sh
```

### Test the model: 
```
sh ./script/evaluate_dml_mobilenet_on_market.sh
```
## Results:
![](https://github.com/Guo-Xiaoqing/Skin-Seg/raw/master/result1.png)
Each row includes the original image, dilated rate map, predictions and ground truth from left to right. Note that red in heat map denotes a larger receptive field.

![](https://github.com/Guo-Xiaoqing/Skin-Seg/raw/master/result2.png)
Examples of complementary network results in comparison with other methods. The ground truth is denoted in black. Results of \cite{ronneberger2015u}, \cite{sarker2018slsdeep}, \cite{yuan2017improving} and ours are denoted in blue, cyan, green, and red, respectively.

## Citation:
If you found Triple ANet helpful for your research, please cite our paper:
```
@inproceedings{guo2020complementary,
  title={Complementary network with adaptive receptive fields for melanoma segmentation},
  author={Guo, Xiaoqing and Chen, Zhen and Yuan, Yixuan},
  booktitle={2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI)},
  pages={2010--2013},
  year={2020},
  organization={IEEE}
}
```

## Questions:
Please contact "xiaoqingguo1128@gmail.com" 
