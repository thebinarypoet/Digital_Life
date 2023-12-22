import os
os.chdir('C:\\experiment\\Ai_project\\Digital_Life_Server-master')
import sys
import time

sys.path.append('TTS/vits')

import soundfile
os.environ["PYTORCH_JIT"] = "0"
import torch
import requests
import TTS.vits.commons as commons
import TTS.vits.utils as utils

from TTS.vits.models import SynthesizerTrn
from TTS.vits.text.symbols import symbols
from TTS.vits.text import text_to_sequence

import logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


class TTService():
    def __init__(self, cfg, model, char, speed):
        logging.info('Initializing TTS Service for %s...' % char)
        self.hps = utils.get_hparams_from_file(cfg)
        self.speed = speed
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model).cuda()
        _ = self.net_g.eval()
        # _ = utils.load_checkpoint(model, self.net_g, None)


    def read(self, text, filename):
        '''
        text = text.replace('~', '！')
        stn_tst = get_text(text, self.hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            audio = self.net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.2, length_scale=self.speed)[0][
                0, 0].data.cpu().float().numpy()
        '''
        # 你的语音合成API的地址
        api_url = "http://127.0.0.1:5000/voice"  # 请替换为你的实际端口号

        # 请求参数
        data = {
            "text": text,
            "model_id": 1,  # 请根据你的实际情况选择模型ID
            "speaker_name": "宵宫",
            "sdp_ratio": 0.2,
            "noise": 0.2,
            "noisew": 0.9,
            "length": 1,
            "language": "ZH",
            "auto_translate": False,
            "auto_split": True,
        }
        #print(data)
        # 发起HTTP POST请求
        response = requests.get(api_url, params=data)
        audio_content = response.content
        with open(filename, "wb") as audio_file:
            audio_file.write(audio_content)
        #print(audio)
        return

    def read_save(self, text, filename, sr):
        stime = time.time()
        self.read(text, filename)
        audio_data, _ = soundfile.read(filename)
        soundfile.write(filename, audio_data, _)
        logging.info('VITS Synth Done, time used %.2f' % (time.time() - stime))




