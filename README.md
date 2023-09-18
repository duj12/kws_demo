# Key Word Spotting

## 使用方法

### 1. 初始化kws, 并配置好唤醒词
```python
from stream_kws_ctc import KeyWordSpotter

kws = KeyWordSpotter(
                     ckpt_path='model/avg_30.pt',
                     config_path='model/config.yaml',
                     token_path='model/tokens.txt',
                     lexicon_path='model/lexicon.txt',
                     threshold=0.02,
                     min_frames=5,
                     max_frames=250,
                     interval_frames=50,
                     score_beam=3,
                     path_beam=20,
                     gpu=-1,
                     is_jit_model=False,)

kws.set_keywords("你好小镜")
```
其中各参数意义如下：
```shell
ckpt_path: 模型路径
config_path: 模型对应配置文件路径
token_path: 字典路径, 
  实际模型可以检测的字符集合, 不在此集合中的字会被检测成此集合同音字
lexicon_path: 词典路径, 
  给出了每个词到字典中基本单元的映射关系, 可以看到一些字被映射成了同音字
threshold: 唤醒词检测阈值, 
  为大于零的一个数值, 数值越大则越不容易检测出来唤醒词(同时越不容易误检)
min_frames: 唤醒词最小的帧数, 
  1帧对应0.01s, 如果检出唤醒词时长低于此帧数, 会直接忽略视为没有唤醒词
max_frames: 唤醒词最大的帧数, 
  1帧对应0.01s, 如果检出唤醒词时长高于此帧数, 会直接忽略视为没有唤醒词
interval_frames: 连续两个唤醒词之间间隔帧数, 
  1帧对应0.01s, 如果第二个唤醒词检出时间距离上一个的时间间隔低于此帧数, 第二个唤醒词被忽略
score_beam: 唤醒词解码时, 每帧候选结果数量限制值, 
  每一帧只留下有限个与唤醒词发音相同的概率值参与解码搜索
path_beam: 唤醒词解码时, 所有时刻总的候选解码路径数量限制值, 
  任何时刻最多保留一定数量的解码路径
gpu: 使用GPU的序号, 
  -1表示不使用GPU, 大于等于0时表示选用机器上存在的某一块GPU
is_jit_model: 是否是JITScript模型, 
  默认提供的都是torch的checkpoint模型
```

### 2. 输入音频
注意，算法只支持16k采样率，16bit的单通道音频输入
```python
import librosa

wav_path = '/path/to/your/wave'

y, _ = librosa.load(wav_path, sr=16000, mono=True)
 # NOTE: model supports 16k sample_rate
wav = (y * (1 << 15)).astype("int16").tobytes()

```

### 3. 流式检测

```python
# We inference every 0.3 seconds, in streaming fashion.
interval = int(0.3 * 16000) * 2
for i in range(0, len(wav), interval):
    chunk_wav = wav[i: min(i + interval, len(wav))]
    result = kws.forward(chunk_wav)
    print(result)
```

### 4. 输出格式
输出格式为dict，包含state, keyword, start, end, score五个元素
```shell
"state": 关键词检测状态, 1表示检出，0表示未检出
"keyword": 检测出的关键词
"start": 关键词起始时间，以此次连接为起始时刻
"end": 关键词结束时间
"score": 关键词的分数，越高则表示关键词置信度越高

如果检测出了关键词，那么结果格式如下
{
    "state": 1,
    "keyword": keyword,
    "start": start,
    "end": end,
    "score": score
}
否则，结果格式如下：
{
    "state": 0,
    "keyword": None,
    "start": None,
    "end": None,
    "score": None
}

```
