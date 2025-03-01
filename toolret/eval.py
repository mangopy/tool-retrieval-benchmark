from typing import Dict, Tuple
from datasets import load_dataset, concatenate_datasets
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import faiss
import pytrec_eval
from typing import List, Union, Tuple
from .encode import encode_data, gritlm_instruction, get_pool
from .config import _QUERY_REPO, _TOOL_REPO, _MODEL, _TASK, _CATEGORY, _FIRST_STAGE
from .utils import write_file
from transformers import (AutoTokenizer, AutoModel)
from sentence_transformers import SentenceTransformer
import torch
import os
import json
from collections import defaultdict
from typing import override


os.environ["TOKENIZERS_PARALLELISM"] = "false"
token = os.getenv("HUGGINGFACE_TOKEN")

class RetModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.tokenizer, self.st = load_model_tokenizer(model_name)

    def encode_queries(self, queries, bs, is_inst=False):
        if 'GritLM' in self.model_name:
            embeddings = []
            for item in queries:
                embedding = self.model.encode(
                    [item['query']],
                    instruction=gritlm_instruction(item['instruction'] if is_inst else ''),
                    batch_size=bs
                )
                embeddings.extend(embedding.tolist())
        elif self.st:
            self.model.default_prompt_name = None
            embeddings = self.model.encode(
                [add_instruction(self.model_name, item['query'], item['instruction'] if is_inst else '') for item in queries],
                batch_size=bs
            )
        else:
            embeddings = encode_data(
                data=[add_instruction(self.model_name, item['query'], item['instruction'] if is_inst else '') for item in queries],
                tokenizer=self.tokenizer,
                model=self.model,
                pooler=get_pool(self.model_name),
                batch_size=bs,
                model_name=self.model_name,
                disable=True
            )
        return embeddings

    def encode_tools(self, tools, bs):
        text = [tool['documentation'] for tool in tools]
        if 'GritLM' in self.model_name:
            embedding = self.model.encode(text, batch_size=bs, instruction=gritlm_instruction(""))
        elif self.st:
            embedding = self.model.encode(text, batch_size=bs, show_progress_bar=True)
        else:
            embedding = encode_data(
                text, self.tokenizer,
                self.model,
                get_pool(self.model_name),
                batch_size=bs,
                model_name=self.model_name
            )
        embedding = np.asarray(embedding, dtype=np.float32)
        return embedding


def trec_eval(qrels: Dict[str, Dict[str, int]],
              results: Dict[str, Dict[str, float]],
              k_values: Tuple[int] = (5, 10, 20)) -> Dict[str, float]:
    ndcg, _map, recall, prec,comp = {}, {}, {},{},{}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        prec[f'Precision@{k}'] = 0.0
        comp[f'Comprehensiveness@{k}'] = 0.0


    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            prec[f'Precision@{k}'] += scores[query_id]["P_" + str(k)]
            comp[f'Comprehensiveness@{k}'] += (scores[query_id]["recall_" + str(k)] == 1)

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 5) for k, v in m.items()}

    ndcg = _normalize(ndcg)
    _map = _normalize(_map)
    recall = _normalize(recall)
    prec = _normalize(prec)
    comp = _normalize(comp)

    all_metrics = {}
    for mt in [ndcg, _map, recall, prec, comp]:
        all_metrics.update(mt)

    return all_metrics

def load_model_tokenizer(model_name: str):
    if model_name in ['e5-mistral-7b-instruct', 'gtr-t5', 'gte-Qwen2-1.5B', "Tool-COLT", 'GritLM']:
        st = True
    else:
        st = False

    if 'GritLM' in model_name:
        # 加载 GritLM 模型
        from gritlm import GritLM
        model = GritLM(model_name, torch_dtype="auto")
        tokenizer = None
    elif st:
        # print('Sentence Transformer: ', model_name)
        model_kwargs = {"torch_dtype": torch.float16, 'token': token} if 'instruct' in model_name else None
        model = SentenceTransformer(model_name, trust_remote_code=True, model_kwargs=model_kwargs)
        if 'Tool-COLT' not in model_name:
            model.max_seq_length = 2048
        tokenizer = None
    else:
        # print('AutoModel: ', model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
        if "NV" in model_name:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
        else:
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True, token=token)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

    return model, tokenizer, st

def validate_task(task: Union[str, None]):
    return task.lower() in _TASK

def validate_category(category: Union[str, None]):
    return category in _CATEGORY

def task_split(task: str):
    if task.lower() == 'all':
        _task = _TASK
    else:
        _task = [t.strip().lower() for t in task.split(',')]
    assert all([validate_task(t) for t in _task])
    return _task

def load_queries(task: str):
    queries = load_dataset(_QUERY_REPO, task, cache_dir="./cache", download_mode="force_redownload")['queries']
    return queries


def load_tools(category: str):
    if category.lower() == 'all':
        _categories = _CATEGORY
    else:
        _categories = [c.strip().lower() for c in category.split(',')]
    assert all([validate_category(c) for c in _categories])
    tools = concatenate_datasets([load_dataset(_TOOL_REPO, c)['tools'] for c in _categories])
    return tools


def add_instruction(model_name, query, instruct=None) -> str:
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


def print_results(output_dict, metrics=['NDCG@1', 'NDCG@5', 'NDCG@10'], report_sub_task=True):
    task_results = defaultdict(list)
    for k, v in output_dict.items():
        v = v[-1]
        task_results[v['task']].append(v)
        try:
            sub_task = k.split('/')[1].split('__')[-2]
            sub_task = f"-- {v['task']}/{sub_task}"
            task_results[sub_task].append(v)
        except:
            pass

    table_data = []
    avg_score = [0 for _ in range(len(metrics))]
    avg_size = [0 for _ in range(len(metrics))]
    for task in task_results:
        line = [task]
        for i, metric in enumerate(metrics):
            score = [x['eval_results'][metric] * x['size'] for x in task_results[task]]
            size = [x['size'] for x in task_results[task]]
            if '-- ' not in task:
                avg_score[i] += sum(score)
                avg_size[i] += sum(size)
            score = sum(score) / sum(size)
            score = score * 100
            if '-- ' in task and not report_sub_task:
                continue
            line.append(f"{score:.2f}")
        table_data.append(line)
    line = ['Avg']
    for i, metric in enumerate(metrics):
        score = avg_score[i] / avg_size[i]
        score = score * 100
        line.append(f"{score:.2f}")
    table_data.append(line)

    headers = ["Data"] + metrics

    try:
        from tabulate import tabulate
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    except ModuleNotFoundError:
        column_widths = [max(len(str(item)) for item in column) for column in zip(headers, *table_data)]
        header_row = " | ".join(f"{headers[i]:^{column_widths[i]}}" for i in range(len(headers)))
        print(f"| {header_row} |")
        separator_row = "-+-".join('-' * column_widths[i] for i in range(len(headers)))
        print(f"{separator_row}")
        for row in table_data:
            row_str = " | ".join(f"{row[i]:^{column_widths[i]}}" for i in range(len(row)))
            print(f"| {row_str} |")


def eval_retrieval(model_name: str,
                   tasks: List,
                   category: str = 'all',
                   batch_size: int = 4,
                   output_file: str = None,
                   top_k: int = 100,
                   is_inst: bool = False,
                   is_print: bool = True):
    # load the model
    model = RetModel(model_name,)

    # encode the tools to embeddings
    tools = load_tools(category)
    print(len(tools))
    tool_embeddings = model.encode_tools(tools, batch_size)
    tool_embeddings = np.asarray(tool_embeddings, dtype=np.float32)
    dim = tool_embeddings.shape[1]

    # MIPS for semantic matching: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#metric_inner_product
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(tool_embeddings)

    collection = {}
    output = {}

    _tasks = task_split(tasks)
    for task in tqdm(_tasks):
        queries = load_queries(task)
        # encode the queries to embeddings
        query_embeddings = model.encode_queries(queries, batch_size, is_inst)
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

        # match the top-k tools via MIPS
        distance, rank = index.search(query_embeddings, top_k)

        results = {}
        for item, rk, ds in zip(queries, rank, distance):
            results[item['id']] = {}
            for r, d in zip(rk, ds):
                results[item['id']][str(tools[int(r)]['id'])] = float(d)
        
        qrels = {}
        for item in queries:
            # if item["labels"]:
            qrels[item['id']] = {str(x['id']): int(x['relevance']) for x in json.loads(item['labels'])}
                
        collection[task] = trec_eval(qrels=qrels, results=results)
        output[task] = results

    if output_file is not None:
        write_file(output, output_file)

    # if is_print:
        # print_results(collection)
    return collection

from typing import override

class RankModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model, _ = self.load_model_tokenizer(model_name)

    def load_model_tokenizer(self, model_name):
        try:
            if 'bge-rerank' in model_name:
                from FlagEmbedding import FlagAutoModel, FlagReranker, FlagLLMReranker  # a python library, please `pip install flagembedding==1.3.3`
                if "gemma" in model_name:
                    model = FlagLLMReranker(model_name, use_fp16=True, batch_size=2)
                else:
                    model = FlagReranker(model_name, use_fp16=True)
            else:
                if "t5" in model_name:
                    model_type = "t5"
                else:
                    model_type = 'cross-encoder'
                from rerankers import Reranker  # a python library, please `pip install rerankers==0.5.0`
                model = Reranker(model_name=model_name, model_type=model_type)
            return model, None
        except Exception as e:
            print(f"load model error: {e}")


    def compute_rank_score(self, query: str, tools: str, instruction: str = ''):
        raise NotImplementedError


class FlagRankModel(RankModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    @override
    def compute_rank_score(self, query: str, tools: List[str], instruction: str = None):
        if instruction:
            query = f"Instruct: {instruction}\nQuery: {query}"
            score = self.model.compute_score([[query, text] for text in tools])
        else:
            score = self.model.compute_score([[query, text] for text in tools])
        return score

class HFRankModel(RankModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    @override
    def compute_rank_score(self, query: str, tools: List[str], instruction: str = None):
        if instruction:
            query = f"Instruct: {instruction}\nQuery: {query}"
            score = self.model.rank(query, tools)
        else:
            score = self.model.rank(query, tools)
        return score


def eval_rerank(model_name,
                tasks: List[str],
                instruct=True,
                first_stage=None):
    if 'bge-rerank' in model_name:
        model = FlagRankModel(model_name)
    else:
        model = HFRankModel(model_name)
    outputs = {}
    for task in tasks:
        queries = load_queries(task)['queries']
        tools = load_dataset(_FIRST_STAGE, task)['tools']
        tools = {item['id']: item['tools'] for item in tools}
        results = defaultdict(lambda : defaultdict(float))

        for item in tqdm(queries):
            candidates = tools[item['id']][:100]
            instruction = None if instruct else item['instruction']
            scores = model.compute_rank_score(query=item['query'],
                                             candidates=[tool['documentation'] for tool in candidates],
                                             instruction=instruction)
            for tool, score in zip(candidates, scores):
                results[item['id']][tool['id']] = float(score)
        outputs[task] = results
    return outputs


def eval_bm25(tasks,
              output_file: str,
              instruct=True,
              rk_num=100,):
    import bm25s
    results = {}
    for task in tasks:
        if task in runs and not args.force_replace:
            continue
        queries = load_queries(task)
        tools = load_tools(task)
        doc_content = [str(item['doc']) for item in tools]
        doc_ids = [item['id'] for item in tools]
        corpus_tokens = bm25s.tokenize(doc_content, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        result = {}
        for item in queries:
            query = item['query']
            if instruct:
                query = item['instruction'] + ' ' + query
            query_tokens = bm25s.tokenize(query)
            if len(query_tokens.vocab) == 0:
                query_tokens = bm25s.tokenize('NONE', stopwords=[])
            hits, scores = retriever.retrieve(query_tokens, corpus=doc_ids, k=min(rk_num, len(doc_ids)))
            result[item['id']] = {}
            for i in range(len(hits[0])):
                result[item['id']][hits[0, i]] = float(scores[0, i])

        qrels = {}
        qrels[item['id']] = {str(x['id']): int(x['relevance']) for x in item['labels']}

        results[task] = result
        write_file(results, output_file)

def eval_colbert_v2(model_name: str,
                    tasks: str,
                    index_dir: str,
                    category: str,
                    output_dir: str,
                    instruct=False,):
    import colbert
    from colbert import Indexer, Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    print("eval colbert start")
    nbits = 2   # encode each dimension with 2 bits
    doc_maxlen = 512 # truncate passages at 512 tokens
    tool_data = load_tools(category)
    tool_doc_data = [str(tool["doc"]) for tool in tool_data]
    os.makedirs(index_dir, exist_ok=True)

    results = {}
    for task in tasks:
        query_data = load_queries(task=task)

        os.makedirs(os.path.join(index_dir, task), exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)
        index_name = os.path.join(index_dir, task, "index.2bits")
        with Run().context(RunConfig(nranks=1, experiment='toolret')):
            config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4)
        indexer = Indexer(checkpoint=model_name, config=config)
        indexer.index(name=index_name, collection=tool_doc_data, overwrite='force_silent_overwrite')

        #search
        with Run().context(RunConfig(experiment='notebook')):
            searcher = Searcher(index=index_name, collection=tool_doc_data)
        for q in tqdm(query_data):# Find the top-3 passages for this query
            query = q['query']
            if instruct:
                query = q['instruction'] + ' ' + query
            qid = q["id"]
            outputs = searcher.search(query, k=100)

            # Print out the top-k retrieved passages
            results[task][qid] = {}
            for passage_id, passage_rank, passage_score in zip(*outputs):
                results[task][qid][tool_data[passage_id]["id"]] = passage_score
        write_file(results, output_dir)
