#!/usr/bin/env python3
"""DSV4.1 absolute-speed benchmark. Dedicated idle server ONLY: resets its cache.

Python standard library. No credentials are saved. See the accompanying report
for workload, token accounting, fixed-output caveats and metric definitions.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.request

LOCK = threading.Lock()
PARAGRAPH = '观测站每天记录温度、湿度、气压和设备状态。工程师先核对原始数据，再比较不同仪器的读数，识别异常点，保留完整的校准记录。长期分析需要区分季节变化、传感器漂移和真实环境变化，不能仅凭一次测量下结论。'
TASK = '请根据这些观测记录，用中文撰写一份详尽的技术手册，依次解释测量原理、误差来源、校准流程、数据质量检查、故障排查、长期维护和具体案例。每部分展开充分，持续写作，不要提前总结，不少于六千字。'

def api(path, body=None):
    raw = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(ARGS.url + path, data=raw, headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=1200)

def metrics():
    with api('/metrics') as r:
        return {k: float(v) for k,v in re.findall(r'^tensorfold:([a-z_]+) ([0-9.]+)$', r.read().decode(), re.M)}

def idle_metrics():
    # SSE completion can precede the engine's published idle snapshot by a tick.
    deadline=time.monotonic()+15
    consecutive=0
    while time.monotonic()<deadline:
        m=metrics()
        consecutive=consecutive+1 if m['requests_running']==m['requests_waiting']==0 else 0
        if consecutive>=2:return m
        time.sleep(.1)
    raise RuntimeError('Dedicated server did not become idle after requests completed')

def body_for(nonce, repetitions, thinking=True, budget=1024):
    return {'model': ARGS.model, 'temperature': 0, 'seed': 123,
            'max_tokens': budget, 'ignore_eos': True,
            'chat_template_kwargs': {'enable_thinking': thinking},
            'messages': [{'role': 'system', 'content': f'独立测量编号 {nonce}。请认真完成用户的技术写作任务。'},
                         {'role': 'user', 'content': PARAGRAPH * repetitions + '\n\n' + TASK}],
            'stream': True, 'stream_options': {'include_usage': True}}

def tokenize(body):
    with api('/tokenize', {'model': ARGS.model, 'messages': body['messages'],
                          'chat_template_kwargs': body['chat_template_kwargs']}) as r:
        data = json.load(r)
        return data['count'], data['tokens']

def fitted(target, nonce):
    # Whole Chinese paragraphs; actual rendered counts are retained, not rounded.
    count0,_ = tokenize(body_for(nonce, 0))
    count1,_ = tokenize(body_for(nonce, 1))
    reps = max(1, round((target-count0)/(count1-count0)))
    for _ in range(5):
        body = body_for(nonce, reps)
        count,ids = tokenize(body)
        if abs(count-target)<150:break
        reps=max(1,round(reps*(target-count0)/(count-count0)))
    assert abs(count-target) < 150, (target,count)
    return body, {'target': target, 'rendered_tokens': count, 'paragraphs': reps,
                  'prompt_token_sha256': hashlib.sha256(json.dumps(ids, separators=(',',':')).encode()).hexdigest()}

def request(label, body, directory):
    raw = json.dumps(body, ensure_ascii=False).encode()
    row = {'label': label, 'started_epoch': time.time(), 'started': time.monotonic(),
           'request_sha256': hashlib.sha256(raw).hexdigest(), 'usage': {}, 'tensorfold': {},
           'delta_times': [], 'text': '', 'reasoning': ''}
    try:
        with api('/v1/chat/completions', body) as response:
            for line in response:
                if not line.startswith(b'data:'): continue
                s = line[5:].strip()
                if s == b'[DONE]': continue
                event = json.loads(s)
                if event.get('error'): raise RuntimeError(event['error'])
                for key in ('usage','tensorfold','speculative'):
                    if event.get(key): row[key] = event[key]
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    text = delta.get('content') or ''
                    thought = delta.get('reasoning_content') or ''
                    if text or thought:
                        row['delta_times'].append(time.monotonic())
                        row['text'] += text
                        row['reasoning'] += thought
                    if choice.get('finish_reason'): row['finish_reason'] = choice['finish_reason']
        assert row['usage'] and row['tensorfold'] and row['delta_times'], 'Missing timing/usage/content'
    except BaseException as exc:
        row['error'] = repr(exc)
        raise
    finally:
        row['finished'] = time.monotonic()
        row['wall_s'] = row['finished'] - row['started']
        row['ttft_s'] = row['delta_times'][0] - row['started'] if row['delta_times'] else None
        row['output_sha256'] = hashlib.sha256((row['reasoning'] + row['text']).encode()).hexdigest()
        row['cached_tokens'] = row['usage'].get('prompt_tokens_details', {}).get('cached_tokens',0)
        with LOCK:
            with (directory/'requests.jsonl').open('a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps({k:row.get(k) for k in ('label','wall_s','ttft_s','usage','tensorfold','error')},ensure_ascii=False), flush=True)
    return row

def reset_cache(directory):
    # Reserve the complete logical KV pool with a tiny, EOS-terminated request.
    # This exercises normal admission; no engine config or private reset API.
    body = body_for('RESET',0,False)
    body['messages'] = [{'role':'user','content':'Reply with exactly OK and nothing else.'}]
    body['ignore_eos'] = False
    n,_ = tokenize(body)
    body['max_tokens'] = ARGS.context - n - 18
    for i in range(2):
        row = request(f'reset-{i}',body,directory)
        assert row['usage']['completion_tokens'] < 128, 'Unexpectedly long reset response'
    m = idle_metrics()
    assert m.get('prompt_cache_states',0) == 0, ('Cache not empty',m)
    assert m['requests_running'] == m['requests_waiting'] == 0

def batch(label, bodies, directory):
    barrier = threading.Barrier(len(bodies)+1)
    def go(i,body):
        barrier.wait()
        return request(f'{label}-s{i+1}',body,directory)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(bodies)) as executor:
        futures = [executor.submit(go,i,b) for i,b in enumerate(bodies)]
        barrier.wait()
        return [f.result() for f in futures]

def run_cell(target, concurrency, repetition):
    label = f'p{target}-c{concurrency}-r{repetition}'
    directory = ARGS.out/label
    if directory.exists():
        if not getattr(ARGS,'resume',False):raise FileExistsError(directory)
        if (directory/'complete.json').exists():
            print('CELL_ALREADY_COMPLETE',label,flush=True)
            return
        directory.rename(ARGS.out/f'incomplete-{label}-{int(time.time())}')
    directory.mkdir()
    before = idle_metrics()
    assert before['requests_running'] == before['requests_waiting'] == 0
    reset_cache(directory)
    fitted_bodies = [fitted(target,f'{ARGS.run_id}-{label}-{i}') for i in range(concurrency)]
    bodies = [item[0] for item in fitted_bodies]
    (directory/'prompts.json').write_text(json.dumps([item[1] for item in fitted_bodies],indent=2))
    samples, done = [], threading.Event()
    def monitor():
        while not done.is_set():
            try: samples.append({'time':time.monotonic(),**metrics()})
            except Exception as exc: samples.append({'error':repr(exc)})
            done.wait(.5)
    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        cold = batch('cold',bodies,directory)
        warm = batch('warm',bodies,directory)
    finally:
        done.set(); thread.join()
        after = idle_metrics()
        (directory/'metrics.json').write_text(json.dumps({'before':before,'after':after,'samples':samples}))
    for i,(c,w) in enumerate(zip(cold,warm)):
        assert c['cached_tokens'] == 0, ('Not cold',label,i)
        assert c['usage']['prompt_tokens'] == fitted_bodies[i][1]['rendered_tokens']
        assert w['cached_tokens'] == (w['usage']['prompt_tokens']-128)//2048*2048, ('Incomplete warm hit',label,i,w['cached_tokens'])
        assert c['request_sha256'] == w['request_sha256']
        assert c['usage']['completion_tokens'] == w['usage']['completion_tokens'] == 1024
    expected = 2 + 2*concurrency
    assert after['request_latency_seconds_count']-before['request_latency_seconds_count'] == expected, 'Other traffic or missing request'
    assert after['prompt_cache_restore_failed_total'] == before['prompt_cache_restore_failed_total'], 'Restore failure'
    assert not any('error' in s for s in samples)
    summary = {'label':label,'target':target,'concurrency':concurrency,'round':repetition,
               'hash_matches':[c['tensorfold']['token_sha']==w['tensorfold']['token_sha'] for c,w in zip(cold,warm)],
               'exclusive':True,'restore_failures':0,'max_running':max(s['requests_running'] for s in samples)}
    (directory/'complete.json').write_text(json.dumps(summary,indent=2))
    print('CELL_COMPLETE',json.dumps(summary),flush=True)

def main():
    ARGS.out.mkdir(parents=True,exist_ok=ARGS.resume)
    settings=ARGS.out/('settings-resume.json' if ARGS.resume else 'settings.json')
    settings.write_text(json.dumps({**vars(ARGS),'out':str(ARGS.out)},indent=2))
    warmup = body_for('WARMUP',40,budget=128)
    request('warmup',warmup,ARGS.out)
    if ARGS.pilot:
        run_cell(4096,1,0)
    else:
        for repetition in range(1,ARGS.rounds+1):
            for target in (4096,32768,190000):
                order = (1,2,3,4) if repetition%2 else (4,3,2,1)
                for concurrency in order: run_cell(target,concurrency,repetition)
    (ARGS.out/'COMPLETE').write_text(str(time.time()))
    print('BENCHMARK_COMPLETE',flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:18892')
    parser.add_argument('--model',default='aya-local')
    parser.add_argument('--context',type=int,default=1048576)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--run-id',default='20261026-absolute')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--resume',action='store_true')
    ARGS=parser.parse_args()
    main()
