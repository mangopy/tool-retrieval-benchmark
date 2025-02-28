import re
import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def gritlm_instruction(instruction):
    instruction = instruction or ""
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

def get_pool(model_name):
    def last_token_pool(last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def average_pool(last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def bos_pool(last_hidden_state, attention_mask):
        return last_hidden_state[:, 0]

    if 'e5-mistral-7b-instruct' in model_name:
        return last_token_pool
    elif 'e5' in model_name:
        return average_pool
    elif 'contriever' in model_name:
        return average_pool
    elif 'all-MiniLM-L6-v2' in model_name:
        return mean_pooling
    elif 'gte' in model_name:
        return bos_pool
    elif 'bge' in model_name:
        return bos_pool
    else:
        return bos_pool


def trunc(sentence, n):
    words = re.finditer(r'\S+|\s+', sentence)
    word_count = 0
    result = []
    for match in words:
        if match.group().strip():
            word_count += 1
        if word_count > n:
            break
        result.append(match.group())
    return ''.join(result)

class EvalData(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item]
        text= str(text) #add
        text = trunc(text, 2048)
        #此时一个item是一个doc的dict需要转化为str
        text_ids = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
        return torch.tensor(text_ids)

    def collate_fn(self, batch):
        input_ids = pad_sequence(batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        features = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return features

@torch.no_grad()
def encode_data(data, tokenizer, model, pooler, batch_size=32, model_name=None, prefix=0, disable=False):
    length_sorted_idx = np.argsort([-len(sen) for sen in data])
    data_sorted = [data[idx] for idx in length_sorted_idx]
    try:
        max_length = min(model.config.max_position_embeddings, 2048)
    except:
        max_length = 2048
    dataset = EvalData(data_sorted, tokenizer, max_length)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, shuffle=False,
                                              batch_size=batch_size, num_workers=8)
    all_embedding = []
    for batch in tqdm(data_loader, disable=disable):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            if model_name is not None and 'NV-Embed' in model_name:
                batch = model.prepare_kwargs_from_batch(batch, prefix, model.device)
                embeddings = model(**batch)['sentence_embeddings'].squeeze(1)
            else:
                outputs = model(**batch)
                embeddings = pooler(outputs.last_hidden_state, batch['attention_mask'])
            if 'gtr-t5' not in model_name and 'contriever' not in model_name:
                embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.cpu()
        all_embedding.extend(embeddings)
    all_embedding = [all_embedding[idx] for idx in np.argsort(length_sorted_idx)]
    all_embedding = np.asarray([emb.numpy() for emb in all_embedding])
    return all_embedding

def get_prompt_name(model_name, instruct=None):
    if 'e5-mistral-7b-instruct' in model_name:
        return 'web_search_query'
    else:
        return None
    

def get_query(model_name, query, instruct=None) -> str:
    if 'e5-mistral-7b-instruct' in model_name:
        task_description = instruct or 'Given a web search query, retrieve relevant passages that answer the query'
        return f'Instruct: {task_description}\nQuery: {query}'
    elif 'NV-Embed-v1' in model_name:
        task_description = instruct or "Given a question, retrieve passages that answer the question"
        return "Instruct: " + task_description + "\nQuery: " + query + '</s>'
    elif 'e5' in model_name:
        task_description = "Instruct: " + instruct + '\n' if instruct is not None else ""
        return task_description + 'query: ' + query
    else:
        task_description = "Instruct: " + instruct + '\nQuery: ' if instruct is not None else ""
        return task_description + query

def get_instruct_length(tokenizer, instruct=None):
    task_description = instruct or "Given a question, retrieve passages that answer the question"
    instruction = "Instruct: " + task_description + "\nQuery: "
    return len(tokenizer.tokenize(instruction))