import torch.nn as nn

class model_head(nn.Module):

    def  __init__(self):
        super().__init__()
        self.activation = nn.BCELoss()
        self.first_layer = nn.Linear(in_features=326400, out_features=100)
        self.second_layer = nn.Linear(in_features=100, out_features=3)



    def forward(self, input):
        first_value = self.activation(self.first_layer(input))
        inference = self.activation(self.second_layer(first_value))
        return inference


