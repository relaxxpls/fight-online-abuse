import torch.nn as nn


class DistilBertClassifier(nn.Module):
    def __init__(self, bert, num_labels):
        super().__init__()

        self.num_labels = num_labels
        self.bert = bert

        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_token = sequence_output[:, 0, :]
        logits = self.classifier(cls_token)

        return logits
