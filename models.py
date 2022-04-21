import torch.nn as nn


class DistilBertClassifier(nn.Module):
    def __init__(self, bert, num_labels):
        super().__init__()

        self.num_labels = num_labels
        self.bert = bert

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_labels),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        x = outputs.last_hidden_state
        x = x[:, 0]
        x = self.classifier(x)

        return x
