## 数字生命后端

### 大文件下载

#### Bert模型

将下列内容下载然后放在TTS/xg文件夹下的Bert下

```
链接：https://pan.baidu.com/s/16UslYliK3Qm6KVARhVHHfg?pwd=ekhf 
提取码：ekhf
```

#### 语音合成模型

将下列内容下载然后放在TTS/xg文件夹下的model下

```
链接：https://pan.baidu.com/s/1hTh6TGtEwqKmqMI3YQ-Ncg?pwd=fup9 
提取码：fup9
```

#### 语音识别模型和情绪识别模型

将语音识别模型放在ASR目录下的models，情绪识别模型放在SentimentEngine目录下的models里

```
链接：https://pan.baidu.com/s/1slRcguf6GEHFu237D5Rtww?pwd=075y 
提取码：075y
```

### 路径更改

- 更改`SocketServer.py`中的路径
- 更改`TTService.py`中的路径
- 更改`Server_fastapi.py`中的路径

### 部署启动

- **使用python版本为3.9，安装依赖**

```py
pip install -r requirements.txt
```

- **启动run-gpt3.5-api.bat**
- **运行xg文件夹下的server_fastapi.py启动语音合成**

### 打包app项目

[DL_Launcher](https://github.com/thebinarypoet/Launcher_DL)

