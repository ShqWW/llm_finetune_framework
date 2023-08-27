# LLM模型finetune框架使用

cuda版本 cuda11.8

操作系统 ubuntu22.04 server

## 新建conda环境并安装包安

### 1.到pytorch官网使用pip安装pytorch最新版

### 2.安装框架所需包
```
pip install -r requirements.txt
```

## 模型与数据

### 1.数据新建并放在`data`文件夹
```
mkdir data
```

### 2.模型在huggingface平台上下载并放在`remote_scripts`文件夹

```
mkdir remote_scripts
```

### 3.各种模型网址
[Baichuan-7B](https://huggingface.co/baichuan-inc/Baichuan-7B)

[Baichuan-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base)

[Qwen-7B](https://huggingface.co/Qwen/Qwen-7B)

[moss-moon-003-base](https://huggingface.co/fnlp/moss-moon-003-base)

[chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)

### 4.模型配置文件放在`config`文件夹下

## 运行

### 1.训练模版`train.sh`
```
sh train.sh
```


### 2.推理模版`infer.sh`
```
sh infer.sh
```
