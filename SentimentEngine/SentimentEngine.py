import logging
import onnxruntime
from transformers import BertTokenizer
import numpy as np


class SentimentEngine():
    def __init__(self, model_path):
        logging.info('Initializing Sentiment Engine...')
        # 加载onnx模型
        onnx_model_path = model_path
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def infer(self, text):
        tokens = self.tokenizer(text, return_tensors="pt")
        input_dict = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
        input_dict["input_ids"] = input_dict["input_ids"].numpy().astype(np.int64)
        input_dict["attention_mask"] = input_dict["attention_mask"].numpy().astype(np.int64)
        logits = self.ort_session.run(["logits"], input_dict)[0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        print(probabilities)
        predicted = np.argmax(probabilities, axis=1)[0]
        logging.info(f'Sentiment Engine Infer: {predicted}')
        return predicted


if __name__ == '__main__':
    t = '好生气啊，你怎么可以这样'
    s = SentimentEngine('models/sentiment_onnx.onnx')
    r = s.infer(t)
    print(r)

