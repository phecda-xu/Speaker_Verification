# GE2Eloss speaker verification

## 目录

- 一、环境配置（environment）
- 二、数据准备（data prepare）
- 三、训练 
- 四、d_vector提取


## 一、环境配置 environment

- virtualenv 建虚拟环境
- tensorflow > 1.13.1
- python3.5
- requirements 安装依赖

```
$ pip install -r requirements.txt
```


## 二、数据准备 data prepare

- example aishell_1

```
目的将数据集中每个人的音频数据（audio）提特征(feature)，然后合并到一个`.npy`文件中保存；
修改run_data.sh，设置--audio_path '' --feature_path ''两个参数的值
$ sh run_data.sh 
```

## 三、训练

```
训练模型,指定音频特征保存的位置和设置训练模式；
修改run_train.sh, 设置 --feature_path '' --train True 两个参数的值
$ sh run_train.sh
```

### 3.1 net

```
网络结构定义在model.py文件中，同时包含了训练函数 train()
```
### 3.2 参数设置

```
configuration.py 设置参数的位置，可以直接修改内容，设定后用python执行相关的程序，也可以在shell脚本中进行具体参数指定；
```

### 3.3 训练数据及测试数据的预处理

```
data_preprocess.py 将音频数据转化为特征数据
```

### 3.4 功能函数实现

```
utils.py
```

## 四、提取d_vector

输入：wavfile 

输出：由每一帧音频提取的d_vector 组成的矩阵

```
vectorizer.py
```
