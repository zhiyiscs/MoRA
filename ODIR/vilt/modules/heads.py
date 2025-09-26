import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class Prompt_Pooler(nn.Module):
    def __init__(self, prompt_num, prompt_length, hidden_size):
        super().__init__()
        self.prompt_num = prompt_num
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size

        self.attn = nn.MultiheadAttention(hidden_size, 16,batch_first=True)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, prompt,hidden_states):
        first_token_tensor = hidden_states[:, 0:1]
        prompt_token = prompt.view(prompt.size(0),self.prompt_num*self.prompt_length,self.hidden_size)

        attn_input = torch.cat([first_token_tensor,prompt_token],dim=1)

        attn_output = self.attn(attn_input,attn_input,attn_input)

        pooled_output = self.dense(attn_output[0][:,0] + hidden_states[:, 0])
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MPPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 256 * 3)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x
