# Optional long verbatim reasoning-loop guard

This ports the long-passage detector from ashhart/TensorFold commits
`e9ba70f4b1c64152bacd287ae34ef5013206d8ad` and allocation-safety follow-up
`0ce409c75623ce6bcab73aadba7c8be2e52a4faa` to this recipe's pinned TP2 engine.
Original detector authorship belongs to BobClawblaw and the TensorFold
maintainers. `long_loop.zig` is byte-identical to current upstream (SHA-256
`51855fac952c4b51e34335bd0174ec4cac5864bc2336669f2d5144d11096ff3a`).
Only integration into the older stream lifecycle is adapted.

## Configuration

Set `LOOP_GUARD=1` in `scripts/local.sh`. Default `0` leaves the existing behavior
unchanged. This passes `--loop-guard` to the head; it does not impose a thinking
budget, remove reasoning history, change client compaction or enable stop-only
behavior. Existing periods 1 through 8 detection stays in place; long verbatim
cycles gain the upstream indexed detector. The index is allocated/reserved
before committing tokens, and released with the stream.

The guard closes reasoning and allows the reply to continue. It is a mitigation
for exact repetition, not a semantic-loop detector, a checkpoint fix or a
promise that the next action is correct. Continuation can still repeat actions;
callers should keep their normal tool and task validation.

## Validation and limits

- Patch applies independently after all 53 patches on recipe `c7f2b851` / engine
  `fbbc971`; it does not require open PRs #115 or #118 or the cache-policy PR.
- `zig build -j2 test --summary all`: all 23 steps succeed, using the pinned
  Linux ARM64 compiler in a CPU-only container, including upstream synthetic
  detector, lifecycle and allocation tests. Successful cached targets were
  reused on the non-root rerun; a permission-fault test must not run as root.
- Recipe script suite: 33/33 pass in a non-root container with the en_US UTF-8
  locale installed. Added launcher regression checks default-off, opt-in,
  invalid configuration and no history-removal or stop-only flag.
- Earlier paired guard off/on/off testing on the retained older engine scored
  160/164 HumanEval, 151/164 HumanEval+ and 12/18 agent in each mode; coding
  samples matched. Non-thinking quality profiles establish serving stability,
  not post-intervention correctness.
- On the latest-Mia deployment with this backport plus separate tool, prefix and
  cache-policy fixes, three real guard interventions (periods 168, 181, 1285)
  matched the prior guarded engine's outputs and proposed distinct next actions.
  Tool acknowledgments were simulated, with full reasoning retained. No
  original task commands were executed, and original task completion is unknown.
  Both engines completed 252 valid deterministic-workflow actions without
  duplicates, with the same count-31 instead of 32 final reporting error.

The live results above are supporting deployment evidence for the stated builds,
not isolated GPU qualification of this PR. Still outstanding for independent
GPU qualification: a native two-Spark build of this branch alone, drafted-versus-plain token
comparison, resumed/concurrent checks and headroom at upstream default context/
concurrency. Existing production was not interrupted for PR preparation. Private
conversation captures are not included; detector tests use synthetic token IDs.

After changing `LOOP_GUARD`, use `./start.sh restart` to apply it to an existing pair; a running pair is not reconfigured in place.
