# What `start.sh` and `scripts/prepare.sh` do



`scripts/prepare.sh` (run by `start.sh` when something is missing; safe to re-run) on the head:

1. checks docker, the GPU, ssh and the link;
2. builds the image from `zig/Dockerfile` (the pinned engine plus the inputs' tools) and copies it to the worker;
3. downloads the checkpoint and Engram's two shards (with their index and config) at their pinned revisions, and checks the
   shards' sizes (`VERIFY_SHA=1` also their sha256, as the model card lists them: the 1017 tables are new, the older
   release's must not be used);
4. fetches the kit at `KIT_REF`, makes its four RoPE tables and the image routing bias from the checkpoint, and checks every
   file against the kit's MANIFEST (the image bias may differ: it is 1017's own);
5. writes Engram's token map and rank 0's weight file from the checkpoint (no GPU), and the folder rank 0 serves (the
   checkpoint's files linked, with `model/chat_template.jinja`: the release ships its chat format as `encoding/encoding.py`,
   which the server cannot run);
6. creates the worker's NFS volume and checks it;
7. copies Engram's tables to the worker's disk (a decode round reads ~70 random 4 KiB rows from them; over NFS that took most
   of a round) and writes rank 1's weight file there from the checkpoint over NFS.

`start.sh` then starts rank 1 on the worker (`tf-dsv41-lanes`, following) and rank 0 here (`tensorfold-native serve`), waits
for the API, and runs a smoke test.
