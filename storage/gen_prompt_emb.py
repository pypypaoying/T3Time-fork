import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

class GenPromptEmb(nn.Module):
    def __init__(
        self,
        data_path = 'FRED',
        model_name = "gpt2",
        device = 'cuda:0',
        input_len = 96,
        d_model = 768,
        layer = 12,
        divide = 'train',
        prompt_batch_size = 8
    ):  
        super(GenPromptEmb, self).__init__()
        self.data_path = data_path
        self.device = device
        self.input_len =  input_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        self.len = self.input_len-1
        self.prompt_batch_size = prompt_batch_size

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def _prepare_prompt(self, input_template, in_data, in_data_mark, i, j):
        # Time series value
        values = in_data[i, :, j].flatten().tolist()
        values_str = ", ".join([str(int(value)) for value in values])

        # Last token
        trends = torch.sum(torch.diff(in_data[i, :, j].flatten()))
        trends_str = f"{trends.item():0f}"
        
        # Date
        if self.data_path in ['FRED', 'ILI', 'exchange_rate', 'Exchange']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d}"
        elif self.data_path in ['ETTh1', 'ETTh2', 'ECL', 'Traffic']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:00"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:00"
        else: # ETTm1, ETTm2, Weather
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:{int(in_data_mark[i,0,5]):02d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:{int(in_data_mark[i,self.len,5]):02d}"

        # Prompt
        in_prompt = input_template.replace("value1, ..., valuen", values_str)
        in_prompt = in_prompt.replace("Trends", trends_str)
        in_prompt = in_prompt.replace("[t1]", start_date).replace("[t2]", end_date)
        # print("in_prompt: ", in_prompt)

        return in_prompt

    def generate_embeddings(self, in_data, in_data_mark):
        input_templates = {
            'FRED': "From [t1] to [t2], the values were value1, ..., valuen every month. The total trend value was Trends",
            'ILI': "From [t1] to [t2], the values were value1, ..., valuen every week. The total trend value was Trends",
            'ETTh1': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            'ETTh2': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            'ECL': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            'Traffic': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            'ETTm1': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
            'ETTm2': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
            'Weather': "From [t1] to [t2], the values were value1, ..., valuen every 10 minutes. The total trend value was Trends",
            'exchange_rate': "From [t1] to [t2], the values were value1, ..., valuen every day. The total trend value was Trends",
            'Exchange': "From [t1] to [t2], the values were value1, ..., valuen every day. The total trend value was Trends",
        }

        input_template = input_templates.get(self.data_path, input_templates['FRED'])
        prompt_metadata = []
        for i in range(len(in_data)):
            for j in range(in_data.shape[2]):
                prompt_metadata.append(
                    (i, j, self._prepare_prompt(input_template, in_data, in_data_mark, i, j))
                )

        output = torch.empty(
            (len(in_data), self.d_model, in_data.shape[2]),
            dtype=torch.float32,
            device="cpu",
        )
        max_length = self.model.config.n_positions

        for start in range(0, len(prompt_metadata), self.prompt_batch_size):
            chunk = prompt_metadata[start:start + self.prompt_batch_size]
            encoded = self.tokenizer(
                [item[2] for item in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
            last_indices = encoded["attention_mask"].sum(dim=1) - 1
            last_embeddings = hidden[
                torch.arange(hidden.shape[0], device=self.device), last_indices
            ].float().cpu()

            for embedding, (i, j, _) in zip(last_embeddings, chunk):
                output[i, :, j] = embedding

        return output
