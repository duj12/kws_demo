# Copyright (c) 2023 Jing Du (thuduj12@163.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gradio as gr
import wave
from stream_kws_ctc import KeyWordSpotter

kws = KeyWordSpotter(ckpt_path='model/nihaoxiaojing/avg_30.pt',
                     config_path='model/nihaoxiaojing/config.yaml',
                     token_path='model/nihaoxiaojing/tokens.txt',
                     lexicon_path='model/nihaoxiaojing/lexicon.txt',
                     threshold=0.02,
                     min_frames=5,
                     max_frames=150,
                     interval_frames=50,
                     score_beam=3,
                     path_beam=20,
                     gpu=-1,
                     is_jit_model=False,)

# actually this could be done in __init__ method, we pull it outside for changing keywords more freely.
kws.set_keywords("你好小镜")


def detection(audio):
    kws.reset_all()
    if audio is None:
        return "Input Error! Please enter one audio!"

    with wave.open(audio, 'rb') as fin:
        assert fin.getnchannels() == 1
        wav = fin.readframes(fin.getnframes())

    # We inference every 0.3 seconds, in streaming fashion.
    interval = int(0.3 * 16000) * 2
    for i in range(0, len(wav), interval):
        chunk_wav = wav[i: min(i + interval, len(wav))]
        result = kws.forward(chunk_wav)

        if 'state' in result and result['state']==1:
            keyword=result['keyword']
            start=result['start']
            end=result['end']
            txt = f'Activated: Detect {keyword} from {start} to {end} second.'
            return txt
    return "Deactivated."


# input
inputs = [
    gr.inputs.Audio(source="microphone", type="filepath", label='Input audio'),
]

output = gr.outputs.Textbox(label="Output Result")

examples = [
    ['examples/chentianbo_ceqianfang_anjing_0001.wav'],
    ['examples/chentianbo_ceqianfang_neiwaizao_0001.wav'],
    ['examples/chentianbo_ceqianfang_neizao_0001.wav'],
    ['examples/chentianbo_ceqianfang_waizao_0001.wav'],
    ['examples/gongqu-4.5_0000.wav'],
    ['examples/neiwaizao-35.5h_0000.wav'],
    ['examples/neizao-4.5h_0000.wav'],
    ['examples/waizao-5.5h_0000.wav'],
]

text = "Key Word Spotting | 关键词检测"

# description
description = (
    "KWS Demo of Nihao_Xiaojing! This is a key word spotting demo supports \'nihaoxiaojing\' detection !"
)

interface = gr.Interface(
    fn=detection,
    inputs=inputs,
    outputs=output,
    title=text,
    description=description,
    examples=examples,
    theme='huggingface',
)

interface.launch(enable_queue=True)