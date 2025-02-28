<div align="center">
   <h1>ToolRet 🔍</h1>
</div>

🔧 Retrieving useful tools from a large-scale toolset is an important step for Large language model (LLMs) in tool learning. This project (AutoTools) contribute to (i) _the first comprehensive tool retrieval benchmark_ to systematically evaluate existing information retrieval (IR) models on tool retrieval tasks; and (ii) a large-scale training dataset to optimize the expertise of IR models on this tool retrieval task.


## News

- **[2025.1.20]** Our [paper]() is submitted in arxiv! See our [paper]() for details.
- **[2024. 12.7 ]** The blog for our work can be accessed by clicking this [link](). 
- **[2024. 12.7 ]** Our code is released on [Github]() and [HuggingFace](). Please click the link for more details. 



## A New Benchmark -- ToolRet
A concrete example for our evaluation dataset.
```json
{
    "id": "apigen_query_5",
    "query": "Given an initial population of 500 bacteria with a growth rate of 0.3 per minute and a doubling time of 20 minutes, what will be the population after 45 minutes?",
    "labels": [
        {
            "id": "apigen_tool_272",
            "doc": {
                "name": "bacterial_growth",
                "description": "Calculates the bacterial population after a given time based on the initial population and growth rate.",
                "parameters": {
                    "initial_population": {
                        "description": "The initial bacterial population.",
                        "type": "int",
                        "default": 20
                    },
                    "growth_rate": {
                        "description": "The growth rate per unit time.",
                        "type": "float",
                        "default": 20
                    },
                    "time": {
                        "description": "The time elapsed.",
                        "type": "float"
                    },
                    "doubling_time": {
                        "description": "The doubling time of the bacteria in minutes. Defaults to 20.",
                        "type": "float, optional"
                    }
                }
            },
            "relevance": 1
        }
    ],
    "instruction": "Given a `bacterial population prediction` task, retrieve tools that calculate population growth by processing parameters such as initial population, growth rate, elapsed time, and doubling time to provide the projected population size."
}
```

> We release part of our benchmark. The full evaluation dataset will be released after the review processing. See the following `Resource` part for details.

## Python Environment
```shell
conda env create -f requirements.yml
```

## Evaluation

### Quick start
In this work, we systematically evaluate a wide range of advanced IR models.
Below, we show an example for evaluation, where we evaluate the embedding model `intfloat/e5-small-v2`.

```python
# evaluate embedding models
from toolret.eval import eval_retrieval
from toolret.config import _MODEL, _TASK
import os


model = 'intfloat/e5-small-v2' # _MODEL[0]
task = ['all']
output_file = ','.join(task)+'.json'
results = eval_retrieval(model_name=model,
                        tasks=task,
                        category='all',
                        output_file=output_file,
                        is_inst=True)
print(results)
```

The code can be also found in `example/embedding.py` for reference.


### Protocol for model evaluation

Our official experiment uses the following provided hyper-parameters for IR model evaluations.

| Model name         | batch size<br/> (for encode documentation or query)                                   |                               Backend APIs <br/>(to load the model)                                |
|:-------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------:|
| Bge                |                                                                                       |                                        Sentence transformer                                        |



## A Large-scale Training dataset -- ToolRet-train


A concrete example in our training dataset.
```txt

```

## 📊 Model Experiments Results

![img.png](./assets/images/results.png)


## Resource

Our benchmark is built by collecting existing well-established datasets. We provide the raw collected datasets and the processed version (e.g., the ToolRet).

1. The raw dataset can be downloaded from the following link.

| Learning task         | Note                                                                                              |       Link       |
|:----------------------|:--------------------------------------------------------------------------------------------------|:----------------:|
| Tool Understanding  | learning to encapsulate tool documentation in natural language into python functions.             | [Google drive](https://drive.google.com/file/d/1uYIwG1Qj0ut7A1mtjlyKVc_leCOa7hv2/view?usp=sharing) |
| Relevance Learning  | learning to select target tools by generating their identifiers, i.e., tool name.                 | [Google drive](https://drive.google.com/file/d/1qhhe3dviPSTynfbkvlxBhF6-Fk1_VaNx/view?usp=sharing) |
| Function Learning   | learning to programmatically use pre-encapsulated tools for task-solving.                         | [Google drive](https://drive.google.com/file/d/1AOcOh1OzvBJI_J0R3G5DWDtGIB4BC8-p/view?usp=sharing) |

2. The evaluation benchmark, e.g., ToolRet, can be downloaded from the following link.

3. The training dataset, e.g., ToolRet-train, can be downloaded from this link.

4. The trained IR models can be downloaded from the following link

## Acknowledgement
We sincerely thank prior work, such as [MAIR](https://github.com/sunnweiwei/MAIR/) and [ToolBench](https://github.com/OpenBMB/ToolBench), which inspire this project or provide strong technique reference.


## Citation
This work
```text
@article{ToolRetrieval,
      title    = {Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models},
      author   = {Zhengliang Shi, Yuhan Wang, Lingyong Yan, Pengjie Ren, Shuaiqiang Wang, Dawei Yin, Zhaochun Ren},
      year     = 2025,
      journal  = {arXiv},
}
```
Our previous work
```text
@inproceedings{autotools,
	title     = {Tool Learning in the Wild: Empowering Language Models as Automatic Tool Agents},
	author    = {Zhengliang Shi, Shen Gao, Lingyong Yan, Yue Feng, Xiuyi Chen, Zhumin Chen, Dawei Yin, Suzan Verberne, Zhaochun Ren},
	year      = 2025,
	booktitle = {WWW}
}
```

