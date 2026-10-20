import json, os, re, struct, sys
# usage (prepare.sh): python3 -I scripts/ablit_relink_stock.py STOCK_DIR OVERLAY_DIR
# The Keys graft (apply_mia29_wo_b_graft.py, CAST_F32) also casts four MTP tensors to F32 for another engine.
# TensorFold reads mtp.2.confidence_head.proj.weight as F16 (families/dsv41/draft.zig: f16ToF32), so an F32 copy is
# misread and the drafter keeps fewer tokens per round. Shards holding no edited wo_b differ from stock only by those
# casts: link them back to stock, then require the head to be F16.
src, dst = sys.argv[1:]
edited = re.compile(r'(^|\.)layers\.(1\d|2\d|3[0-5])\.attn\.wo_b\.')
conf = 'mtp.2.confidence_head.proj.weight'

def header(path):
    with open(path, 'rb') as f:
        return json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))

relinked = 0
for name in sorted(os.listdir(dst)):
    if not name.endswith('.safetensors'):
        continue
    s, d = os.path.realpath(os.path.join(src, name)), os.path.join(dst, name)
    if os.stat(s).st_ino == os.stat(d).st_ino or any(edited.search(k) for k in header(d)):
        continue
    os.unlink(d)
    os.link(s, d)
    relinked += 1
shard = json.load(open(os.path.join(dst, 'model.safetensors.index.json')))['weight_map'][conf]
dtype = header(os.path.join(dst, shard))[conf]['dtype']
if dtype != 'F16':
    sys.exit(f'{conf} is {dtype}: TensorFold reads it as F16')
print(f'{relinked} shard(s) with no edited wo_b linked back to stock; {conf} is F16')
