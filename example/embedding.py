# evaluate embedding models

from toolret.eval import eval_retrieval
from toolret.config import _MODEL, _TASK
import os
os.environ['HTTP_PROXY'] = 'http://agent.baidu.com:8891'
os.environ['HTTPS_PROXY'] = 'http://agent.baidu.com:8891'

model = _MODEL[0]
print(model)

task = ['all']
output_file = ','.join(task)+'.json'
results = eval_retrieval(model_name=model,
                        tasks=task,
                        category='all',
                        output_file=output_file,
                        is_inst=True)
print(results)