import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from final_head_model import model_head
import torch.nn as nn
from torch.optim import Adam


class train_model():

    def __init__(self):
        self.model_name = "microsoft/deberta-v3-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32  
        )
        self.dataset = pd.read_csv('dataset.csv').reset_index(drop=True)
        self.end_model = model_head()
        self.optimizer = torch.optim.AdamW([
            {"params": self.model.parameters(), "lr": 2e-5},
            {"params": self.end_model.parameters(), "lr": 1e-4},
        ])
        self.loss_fn = nn.MSELoss()




    
    def train(self, epochs):
        for epoch in range(0, epochs):
            self.dataset.sample(1).reset_index(drop=True)
            for i in range(0, len(self.dataset)):

                # zero gradients
                self.optimizer.zero_grad()

                # calculate predictiom
                ground_truth = torch.Tensor([self.dataset.loc[i, 'content'], self.dataset.loc[i, 'organization'], self.dataset.loc[i, 'language']])
                tokens = self.tokenizer(
                            self.dataset.loc[i, 'essay'],
                            truncation=True,
                            padding=True,
                            return_tensors="pt"
                        )
                BERT_prediction = self.model(**tokens)
                flattened_value = BERT_prediction.last_hidden_state[0].flatten()
                prediction = self.end_model(flattened_value)

                # calculate loss and backpropagate
                loss = self.loss_fn(prediction, ground_truth)
                loss.backwards()
                self.optimizer.step()
                

                



obj = train_model()
obj.train(1)
# criteria is: 'content', 'organization', 'language'