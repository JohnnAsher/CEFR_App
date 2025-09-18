import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        logits = outputs.logits / self.temperature
        return SequenceClassifierOutput(
            loss=outputs.loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

class ModelWithVectorTemp(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.temperatures = nn.Parameter(torch.ones(num_classes))

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        logits = outputs.logits / self.temperatures.unsqueeze(0)
        return SequenceClassifierOutput(
            loss=outputs.loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
