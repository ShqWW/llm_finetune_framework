## 新建conda环境并安装包安装包

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

### 1.模型在huggingface平台上下载并放在`remote_scripts`文件夹

```
mkdir remote_scripts
```

## 运行

### 1.训练模版`train.sh`
```
sh train.sh
```


### 2.推理模版`infer.sh`
```
sh infer.sh
```
