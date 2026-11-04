# Podman instead of Docker

The recipe runs under either runtime. Docker stays the default; `CONTAINER_RT=podman` sends every call through podman:
the image build, the two rank containers, the volumes (Engram and NFS), the log handling, the memory guard and the
publishing helper. The CLI calls are the same ones, so most of this page is about the three places where podman is not
a drop-in.

## Settings

Put these in `scripts/local.sh` (or export them):

```sh
CONTAINER_RT=podman
CONTAINER_RT_SUDO=1                 # rootful podman: "sudo -n podman"
# CGROUP_PARENT=tensorfold.slice    # optional; see below
```

- `CONTAINER_RT=docker` (default) — unchanged behaviour, no podman needed.
- `CONTAINER_RT=podman`, `CONTAINER_RT_SUDO=0` — rootless podman as the login user.
- `CONTAINER_RT=podman`, `CONTAINER_RT_SUDO=1` — rootful podman through passwordless `sudo -n`.

**Use rootful podman on a Spark pair.** Two parts of the recipe need root's privileges:

- the worker's read-only NFS volume (`--driver local --opt type=nfs`): a rootless container cannot mount NFS, so
  `prepare.sh`'s file check and rank 1's `/hf` would fail;
- `CGROUP_PARENT`: a system slice (below) is root's; rootless podman can only use the user session's slices.

Rootful podman also matches what the recipe did under Docker: the daemon was root, the ranks ran as root, and the
weight files, Engram copies and spill folders keep the same ownership.

## The three differences

1. **Builds need the host's network.** Podman's build container has its own network namespace. On a Spark the host
   resolves names through systemd-resolved's `127.0.0.53` stub, which that namespace cannot reach, so the Dockerfile's
   first download fails with a name-resolution error. `scripts/runtime.sh` adds `--network host` to `podman build`
   (`rt_build_net`); Docker's build already uses the host's network and gets nothing extra.

2. **Image archives use the docker format.** `prepare.sh` copies the image to the worker with `save | load`. Podman's
   default archive is an OCI archive; `rt_save` writes `--format docker-archive` instead, so the stream also loads on a
   pair that is still half on Docker during a migration.

3. **No `--shm-size` with `--ipc=host`.** The ranks use `--ipc=host`, so the flag is never passed (Docker only ignores
   it). Podman refuses the combination outright; the recipe does not set it, so nothing changes — but do not add it.

Two more that are not differences but are worth knowing:

- `--log-driver json-file` with `--log-opt max-size` / `max-file` works under podman 6; `stop.sh` reads the container
  log the same way. On older podman (4.x) the driver is `k8s-file` and does not rotate by size: leave `LOG_MAX_SIZE`
  empty there, or accept an unrotated log.
- `--gpus all` needs podman 4.9 or later (CDI). `podman --version` on the Sparks is 6.1.

## Cgroup parent (no swap)

The host tuning for these Sparks makes `tensorfold.slice` with `MemorySwapMax=0`. Rootful podman runs containers in
`system.slice` by default, which would drop that guarantee. Set:

```sh
CGROUP_PARENT=tensorfold.slice
```

and `start.sh` adds `--cgroup-parent tensorfold.slice` to both ranks. The slice must exist (it does after the host
tuning); a missing slice makes the rank fail to start, which is the loud failure you want.

## Moving a pair from Docker to Podman

Podman keeps its own image store, so the image has to be copied once per Spark:

```sh
# on the head, from the Docker store into rootful podman's store
sudo -n docker save tensorfold-dsv41:exl3-tp2-<hash> | sudo -n podman load

# then let the recipe do the worker: it compares image identities and copies when they differ
CONTAINER_RT=podman CONTAINER_RT_SUDO=1 ./scripts/prepare.sh
```

Then stop the Docker ranks and start the podman ones:

```sh
./stop.sh                                   # still the docker runtime: it must read CONTAINER_RT from local.sh
CONTAINER_RT=podman CONTAINER_RT_SUDO=1 ./start.sh
```

With the settings in `local.sh`, plain `./stop.sh` and `./start.sh` use podman. The old Docker container and image can
stay until the podman run has been through a restart; `docker ps` should be empty when the podman ranks are up.

Rollback is the same in reverse: remove `CONTAINER_RT` from `local.sh`, `./stop.sh`, `./start.sh`; the Docker image and
container name are untouched by the podman run (different stores, same name).

## Checking it

```sh
sudo -n podman ps --format '{{.Names}} {{.Status}}'     # dsv41-1017-exl3-tf, both Sparks
sudo -n podman inspect -f '{{.State.Running}}' dsv41-1017-exl3-tf
sudo -n podman image inspect -f '{{index .Config.Labels "tf.patches"}}' tensorfold-dsv41:exl3-tp2-<hash>
curl -s localhost:8888/v1/models
```

`DRY_RUN=1 ./start.sh` prints the exact podman command lines for both ranks before anything is started.

## What is not covered

- `scripts/publish-image.sh` pushes with `podman login --authfile` / `podman push`; the registry side is the same.
- The recipe's optional systemd unit (`systemd/dsv41-tensorfold.service`) needs no change: it runs `./start.sh`, which
  reads the settings. Put `CONTAINER_RT`/`CONTAINER_RT_SUDO`/`CGROUP_PARENT` in `scripts/local.sh`, not in the unit,
  so the manual commands and the service agree.
- Rootless podman is not supported for the pair (see above); it works for the image-only helpers (`prepare.sh`'s
  `in_hf` container runs, the kit and token-map steps).
