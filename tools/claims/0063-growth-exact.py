#!/usr/bin/env python3
"""Compare a cold long request with the same request resumed under placement pressure.

Run on an otherwise idle, unsharded server with CONTEXT=524288 and POOL unset.
First start with SPILL_GIB=0 and run `--phase cold`. Restart with spill enabled,
then run `--phase pressure`. Use the same new --seed for both phases. The
pressure phase creates a prefix and a competing tail, and fails unless the
long request actually resumes. Save server logs to verify placement separately.

  python3 -I tools/claims/0063-growth-exact.py --phase cold --scenario fork --seed 12345 --out cold.json
  python3 -I tools/claims/0063-growth-exact.py --phase pressure --scenario fork --seed 12345 --reference cold.json --out pressure.json

A cached-token count alone does not prove that the in-place fallback ran;
confirm it separately with an instrumented engine or a placement regression.
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.request

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from client import API_URL, URL, open_url, prose


def drain_writes():
    start = time.monotonic()
    while True:
        health = json.load(open_url(API_URL + '/health', 30))
        spill = health.get('prompt_cache', {}).get('spill')
        if (not isinstance(spill, list) or len(spill) != 2 or
                any(not isinstance(s, dict) or type(s.get('rank')) is not int or
                    type(s.get('writes_pending')) is not int or s['writes_pending'] < 0
                    for s in spill) or {s['rank'] for s in spill} != {0, 1}):
            raise SystemExit('missing or invalid spill write counters for both ranks')
        if not any(s['writes_pending'] for s in spill):
            return
        if time.monotonic() - start > 60:
            raise SystemExit('spill writes did not finish within 60 seconds')
        time.sleep(.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=('cold', 'pressure'), required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--scenario', choices=('tail', 'fork'), default='tail')
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--reference', type=pathlib.Path)
    args = ap.parse_args()
    if args.phase == 'pressure' and not args.reference:
        ap.error('pressure requires --reference')
    system = {'role': 'system', 'content': 'Answer the final question with one concise sentence.'}
    material = prose(360000, args.seed)
    prefix = 'Material:\n' + material
    question = '\nQuestion: Name three distinct words found in the material.'
    messages = [system, {'role': 'user', 'content': prefix +
                         ('\nBranch B:\n' + prose(1000, args.seed + 2) if args.scenario == 'fork' else '') + question}]
    results = []

    def request(label, msgs, maximum):
        body = {'model': 'DeepSeek-V4.1-Flash-1017', 'messages': msgs,
                'max_tokens': maximum, 'temperature': 0, 'seed': 12345,
                'chat_template_kwargs': {'enable_thinking': False}}
        req = urllib.request.Request(URL, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        start = time.monotonic()
        reply = json.load(open_url(req, 1800))
        usage = reply['usage']
        result = {'label': label, 'seconds': round(time.monotonic() - start, 3),
                  'prompt': usage['prompt_tokens'],
                  'cached': (usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
                  'completion_tokens': usage['completion_tokens'],
                  'message': reply['choices'][0]['message'],
                  'finish_reason': reply['choices'][0]['finish_reason'],
                  'token_sha': reply.get('tensorfold', {}).get('token_sha')}
        results.append(result)
        args.out.write_text(json.dumps({'phase': args.phase, 'scenario': args.scenario, 'seed': args.seed, 'results': results}, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return result

    if args.phase == 'pressure':
        reference = json.loads(args.reference.read_text())
        if reference['phase'] != 'cold' or reference['seed'] != args.seed or reference.get('scenario', 'tail') != args.scenario:
            raise SystemExit('reference phase or seed mismatch')
        seed = [system, {'role': 'user', 'content': prefix + question}] if args.scenario == 'fork' else messages
        seeded = request('seed', seed, 8)
        drain_writes()
        if args.scenario == 'fork':
            fork = [system, {'role': 'user', 'content': prefix + '\nBranch A:\n' + prose(20000, args.seed + 1) + question}]
            branch = request('fork-a', fork, 8)
            if branch['cached'] is None or branch['cached'] < (seeded['prompt'] - 1) // 2048 * 2048:
                raise SystemExit('fork A did not reuse the shared prefix')
        else:
            request('tail', [system, {'role': 'user', 'content': prose(50000, args.seed + 1)}], 8)
        drain_writes()
    target = request('target', messages, 131072)
    if target['prompt'] <= 262144 or target['cached'] is None:
        raise SystemExit('missing cache count or prompt too short')
    if args.phase == 'cold':
        if target['cached'] != 0:
            raise SystemExit('reference was not cold')
    else:
        cold = reference['results'][-1]
        if target['cached'] / target['prompt'] < .95:
            raise SystemExit('pressure request did not resume the long prefix')
        for field in ('prompt', 'completion_tokens', 'message', 'finish_reason'):
            if target[field] != cold[field]:
                raise SystemExit('cold and resumed differ: ' + field)
        if not target['token_sha'] or not cold['token_sha']:
            raise SystemExit('token hash unavailable; text matched but token identity is unproven')
        if target['token_sha'] != cold['token_sha']:
            raise SystemExit('cold and resumed token hashes differ')
        if args.scenario == 'fork':
            drain_writes()
            earlier = results[1]
            restored = request('fork-a-restored', fork, 8)
            expected = (earlier['prompt'] - 1) // 2048 * 2048
            if restored['cached'] is None or restored['cached'] < expected:
                raise SystemExit('fork A did not recover its longer point')
            for field in ('prompt', 'completion_tokens', 'message', 'finish_reason', 'token_sha'):
                if restored[field] != earlier[field] or (field == 'token_sha' and not restored[field]):
                    raise SystemExit('fork A differs after restore: ' + field)
    print('PASS', flush=True)


if __name__ == '__main__':
    main()
