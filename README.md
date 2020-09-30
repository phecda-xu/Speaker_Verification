# GE2Eloss speaker verification

## column

- environment
- data prepare
- train and d_vector extract
- experiments


## environment

- virtualenv
- tensorflow > 1.13.1
- python3.5
- require

```
$ pip install -r requirements.txt
```


## data prepare

- example aishell_1

```
$ python3.5 data_preprocess.py
```

## 训练

TODO

### net

```
model.py
```
### 参数设置

```
configuration.py
```

### 训练及测试数据预处理

```
data_preprocess.py
```

### 功能函数实现

```
utils.py
```

## 提取d_vector

输入：wavfile 

输出：由每一帧音频提取的d_vector 组成的矩阵

```
vectorizer.py
```
