"""Same benchmark, 32K input, thinking OFF, two rounds of concurrency 1..4."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import benchmark as b

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('out',type=Path)
parser.add_argument('--url',default='http://127.0.0.1:18892')
parser.add_argument('--model',default='aya-local')
parser.add_argument('--context',type=int,default=1048576)
args=parser.parse_args()
root=args.out
root.mkdir(parents=True,exist_ok=False)
b.ARGS=SimpleNamespace(url=args.url,model=args.model,context=args.context,
                       out=root,rounds=2,run_id='20261026-answer')
original=b.body_for
def body_for(nonce,repetitions,thinking=False,budget=1024):
    return original(nonce,repetitions,thinking,budget)
b.body_for=body_for
(root/'settings.json').write_text(json.dumps({**vars(b.ARGS),'out':str(root),'thinking':False},indent=2))
b.request('warmup',body_for('WARMUP',40,budget=128),root)
for repetition in (1,2):
    for concurrency in ((1,2,3,4) if repetition==1 else (4,3,2,1)):
        b.run_cell(32768,concurrency,repetition)
(root/'COMPLETE').write_text(str(time.time()))
print('BENCHMARK_COMPLETE',flush=True)
