# Complementary Network with Adaptive Receptive Fields for Melanoma Segmentation

by [Xiaoqing Guo](https://guo-xiaoqing.github.io/), [Zhen Chen](https://franciszchen.github.io/), [Yixuan Yuan](http://www.ee.cityu.edu.hk/~yxyuan/people/people.htm).

## Summary:
### Intoduction:
This repository is for our ISBI2020 paper ["Complementary Network with Adaptive Receptive Fields for Melanoma Segmentation"](https://pdf)
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
Use "make_txt.py" to split training data and testing data. The generated txt files are showed in folder "./txt/".
"make_tfrecords.py" is used to make tfrecord format data, which could be stored in folder "./tfrecord/".

### Train the model: 
```
python3 Triple_ANet_train.py --tfdata_path ./tfrecord/
```

### Test the model: 
```
python3 Triple_ANet_test.py --tfdata_path ./tfrecord/
```
## Results:
![](https://github.com/Guo-Xiaoqing/Skin-Seg/raw/master/result1.png)
Each row includes the original image, dilated rate map, predictions and ground truth from left to right. Note that red in heat map denotes a larger receptive field.

![](https://github.com/Guo-Xiaoqing/Skin-Seg/raw/master/result2.png)
Examples of complementary network results in comparison with other methods. The ground truth is denoted in black. Results of \cite{ronneberger2015u}, \cite{sarker2018slsdeep}, \cite{yuan2017improving} and ours are denoted in blue, cyan, green, and red, respectively.

## Citation:
To be updated

## Questions:
Please contact "xiaoqingguo1128@gmail.com" 
