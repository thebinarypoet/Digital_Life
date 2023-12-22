import logging
import os.path

import torch
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
import torch.onnx


def read_data():
    df = pd.read_csv('datas/bert-data.csv')
    return df['texts'].tolist(), df['labels'].tolist()


class SentimentClassifier():
    def __init__(self, model_path, num_labels=5, max_len=128, batch_size=32, epochs=3, learning_rate=5e-5,
                 scheduler_step_size=1, scheduler_gamma=0.5):
        logging.info('Initializing Sentiment Classifier...')
        self.model_path = model_path
        self.num_labels = num_labels
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertForSequenceClassification(
            BertConfig.from_pretrained('bert-base-chinese', num_labels=self.num_labels)
        ).to(self.device)

    def preprocess_data(self, texts, labels):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        labels = torch.tensor(labels)
        return TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

    def train(self, train_texts, train_labels):
        train_dataset = self.preprocess_data(train_texts, train_labels)

        # 划分训练集和验证集
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # 创建数据加载器
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for batch in train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = criterion(outputs.logits, inputs['labels'])
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            scheduler.step()    # lr优化器

            avg_train_loss = total_loss / len(train_dataloader)
            logging.info(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_train_loss}")

            # 验证
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = tuple(t.to(self.device) for t in batch)
                    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                    outputs = self.model(**inputs)
                    loss = criterion(outputs.logits, inputs['labels'])
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            logging.info(f"Epoch {epoch+1}/{self.epochs}, Validation Loss: {avg_val_loss}")

        logging.info("Training complete!")

        # 保存为onnx模型
        onnx_model_path = os.path.join(self.model_path, 'sentiment_onnx.onnx')
        self.save_model(onnx_model_path)

        logging.info(f"Model saved to {self.model_path}")

    def save_model(self, save_path):
        input_names = ["input_ids", "attention_mask"]
        output_names = ["logits"]
        dynamic_axes = {"input_ids": {0: "batch_size", 1: "sequence_length"},
                        "attention_mask": {0: "batch_size", 1: "sequence_length"}}

        torch.onnx.export(
            self.model,
            (torch.zeros(1, self.max_len).long().to(self.device), torch.zeros(1, self.max_len).long().to(self.device)),
            save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=True,
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    train_texts, train_labels = read_data()

    classifier = SentimentClassifier(model_path='models/')
    classifier.train(train_texts, train_labels)
