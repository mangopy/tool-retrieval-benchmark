
_QUERY_REPO = "mangopy/ToolRet-Queries"
_TOOL_REPO = "mangopy/ToolRet-Tools"
_FIRST_STAGE = 'ToolRet-retrieval-nv-embedd-v1'
_TASK_2_CATEGORY = {
  "craft-math-algebra" : "code",
  "craft-tabmwp" : "code",
  "craft-vqa" : "code",
  "gorilla-huggingface" : "code",
  "gorilla-pytorch" : "code",
  "gorilla-tensor" : "code",
  "toolink" : "code",
  "apibank" : "web",
  "apigen" : "web",
  "mnms" : "web",
  "reversechain" : "web",
  "rotbench" : "web",
  "t-eval-dialog" : "web",
  "t-eval-step" : "web",
  "taskbench-daily" : "web",
  "toolace" : "web",
  "toolbench" : "web",
  "toolemu" : "web",
  "tooleyes" : "web",
  "toollens" : "web",
  "ultratool" : "web",
  "autotools-food" : "web",
  "autotools-music" : "web",
  "autotools-weather" : "web",
  "restgpt-spotify" : "web",
  "restgpt-tmdb" : "web",
  "appbench" : "customized",
  "gpt4tools" : "customized",
  "gta" : "customized",
  "taskbench-huggingface" : "customized",
  "taskbench-multimedia" : "customized",
  "metatool" : "customized",
  "tool-be-honest" : "customized",
  "toolalpaca" : "customized",
  "toolbench-sam" : "customized"
}
_CATEGORY = list(set(_TASK_2_CATEGORY.values()))
_TASK = list(set(_TASK_2_CATEGORY.keys()))

_EMBEDDING_MODEL = [
    "intfloat/e5-small-v2",
    "facebook/contriever-msmarco",
    "sentence-transformers/gtr-t5-base",
    "sentence-transformers/gtr-t5-large",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2", 
    "Alibaba-NLP/gte-base-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-mistral-7b-instruct",
    "Alibaba-NLP/gte-large-en-v1.5",
    "bzantium/NV-Embed-v1",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "GritLM/GritLM-7B",
    "Tool-COLT/contriever-base-msmarco-v1-ToolBenchG3"
]

_RERANKING_MODEL = [
  "castorini/monot5-base-msmarco",
  "jinaai/jina-reranker-v2-base-multilingual",
  "BAAI/bge-reranker-v2-gemma",
  "BAAI/bge-reranker-v2-m3",
  "mixedbread-ai/mxbai-rerank-large-v1"
]

_MODEL = _EMBEDDING_MODEL + _RERANKING_MODEL
