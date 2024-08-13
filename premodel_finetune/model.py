from torch import nn
from transformers import AutoConfig
import torch
from args import args
from transformers import BertPreTrainedModel, BertModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class BertForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.post_init()
    
    def forward(self, x):
        bert_output = self.bert(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits


if __name__ == '__main__':
    config = AutoConfig.from_pretrained("bert-base-chinese")
    model = BertForPairwiseCLS.from_pretrained("bert-base-chinese", config=config).to(device)
    print(model)