---
# 详细文档见https://modelscope.cn/docs/%E5%88%9B%E7%A9%BA%E9%97%B4%E5%8D%A1%E7%89%87
domain: #领域：cv/nlp/audio/multi-modal/AutoML
# - cv
tags: #自定义标签
- 
datasets: #关联数据集
  evaluation: 
  #- damotest/beans
  test:
  #- damotest/squad
  train:
  #- modelscope/coco_2014_caption
models: #关联模型
#- damo/speech_charctc_kws_phone-xiaoyunxiaoyun
deployspec: #部署配置，默认上限GPU4核、内存8GB、无GPU、单实例，超过此规格请联系管理员配置才能生效
# 部署启动文件(若SDK为Gradio/Streamlit，默认为app.py, 若为Static HTML, 默认为index.html)
# entry_file: 
# CPU核数
  cpu: 2
# 内存（单位MB)
  memory: 8000
# gpu个数
  gpu: 0
# gpu共享显存（单位GB，当gpu=0时生效，当gpu>0时显存独占，此配置不生效）
  gpu_memory: 0
# 实例数
  instance: 1
license: Apache License 2.0
---
#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/studios/thuduj12/KWS_Nihao_Xiaojing.git
```