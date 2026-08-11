# Deploying to a public host

Target: a single x86_64 VPS on [Vultr](https://console.vultr.com/), running
`docker compose`. **Cloud Compute** `vc2-4c-8gb` (4 shared vCPU, 8GB RAM,
160GB SSD, Intel, $40/mo) is the closest match to this stack's needs; if
per-request latency matters more than cost, `vhp-4c-8gb-intel` or
`vhp-4c-8gb-amd` (**High Performance** line, dedicated-class vCPU, $48/mo)
trades $8/mo for more consistent CPU. Either is sized well above what this
stack needs at its default `EVENTS_PER_SECOND=8`.

Vultr has no ARM64/Ampere offering — checked directly against its plans API
(`cpu_vendor` is only ever `Intel`, `AMD`, or GPU, across all 151 current
plans), not assumed — so this deploys as **x86_64**, unlike the ARM64 target
in an earlier draft of this doc. That's a non-issue for this stack: every
image in `docker-compose.yml` (`redpandadata/redpanda`, `redpandadata/console`,
`redis:7.4-alpine`, `postgres:16-alpine`, `prom/prometheus`,
`grafana/grafana`) publishes an official `linux/amd64` manifest (verified via
`docker buildx imagetools inspect` for all six), and the five custom images
(`simulator`, `consumer`, `sink`, `api`, `dashboard`) build from
`python:3.12-slim` — amd64 is in fact the more conventionally tested target
for LightGBM/PyTorch/`confluent-kafka` wheels than arm64 was. No image
changes needed.

## 1. Provision

At [console.vultr.com](https://console.vultr.com/): **Products → Deploy New
Server → Cloud Compute**, pick a region close to your users, image
**Ubuntu 24.04 LTS x64**, plan `vc2-4c-8gb` (or `vhp-4c-8gb-*`, see above),
add your SSH key under **SSH Keys** before deploying so it's injected at
boot, then **Deploy Now**.

- Minimum disk: 40GB — the smallest plan above already clears this by a
  wide margin. See [§5](#5-disk-the-simulator-never-stops) before picking
  retention settings; the simulator produces continuously, so disk is the
  one resource that grows unbounded if left on defaults.
- Optional but recommended, done in the console before or right after
  deploy: **Products → Firewall → Add Firewall Group**, allow inbound
  22/tcp, 80/tcp, 443/tcp only, then attach the group to this instance
  under its **Settings → Firewall**. This filters traffic at Vultr's
  network edge, before it reaches the VM at all — a stronger first layer
  than the host-level `ufw` in [§3](#3-firewall-and-ssh-hardening-required-for-any-host-left-running),
  which is worth keeping too but is no longer the only thing standing
  between an accidentally-exposed port and the internet.

```bash
ssh root@<host>
adduser rtrec && usermod -aG sudo rtrec
rsync --archive --chown=rtrec:rtrec ~/.ssh /home/rtrec   # without this rtrec has no
                                                           # SSH key and you're locked out
# re-login as rtrec from here on
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # re-login again to pick this up
```

## 2. Clone and configure

```bash
git clone <repo-url> ~/realtime-recsys && cd ~/realtime-recsys
cp .env.example .env
```

Edit `.env` before first boot:

- `POSTGRES_PASSWORD` — the example value (`rtrec`) is a dev default, not a
  secret. Generate a real one (`openssl rand -hex 24`) anyway: Postgres isn't
  reachable from outside the host at all (`docker-compose.yml` uses `expose:`,
  not `ports:`, for it — see [§3](#3-firewall-and-ssh-hardening-required-for-any-host-left-running)), so the password is defense
  in depth for anyone already inside the host or the compose network, not the
  thing standing between the internet and the offline store.
- `SINK_RETENTION_DAYS` — lower from the dev default of 30 to **7** on a
  small disk. See [§5](#5-disk-the-simulator-never-stops) for the math.

```bash
make dev      # boots everything; drop the profile if you don't want
              # Console/Prometheus/Grafana public (see §4)
make topics
make backfill-partitions
```

`docker-compose.yml` pins `name: rtrec`, so this only matters if you ever run
a second checkout on the same host — it would share this one's volumes
instead of starting empty. Not a concern for a single-checkout VPS.

## 3. Firewall and SSH hardening (required for any host left running)

`ufw` here is not optional or "nice to have for defense in depth" once this
host is up for more than a quick test — treat it as a required step, done
before you walk away. In practice the API's own logs showed automated
scanners hitting it within minutes of boot; a host with an SSH port and a
sudo-capable password open to the internet gets probed continuously, not
occasionally.

That said, `ufw` isn't the only control, and shouldn't be the only thing you
rely on: `docker-compose.yml` already keeps most of the stack off the
internet regardless of the firewall — `postgres` and `redis` use `expose:`
(compose-internal network only, no host port at all), and `api`, `redpanda`,
`console`, `prometheus`, and `grafana`/`dashboard` (the latter two only start
with the `dev` profile) publish to `127.0.0.1:<port>:<port>`, not `0.0.0.0`.
`ufw` closes the remaining gap: SSH itself, which is always reachable by
design and is exactly what those scanners are testing.

```bash
sudo ufw default deny incoming
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

Only 80 and 443 (nginx, [§4](#4-optional-reverse-proxy-and-tls-for-a-public-demo),
optional) need to be open to the internet. Everything else is reached over
an SSH tunnel:

```bash
ssh -L 3000:localhost:3000 -L 8501:localhost:8501 -L 8080:localhost:8080 \
    -L 9090:localhost:9090 rtrec@<host>
# then open http://localhost:<port> locally for Grafana / Streamlit / Redpanda Console / Prometheus
```

**SSH hardening — also required, not optional, for a host left running.**
`adduser rtrec` (§1) creates that user with a sudo-capable password, which is
otherwise brute-forceable on port 22 forever. Disable password auth once key
login is confirmed working:

```bash
# Ubuntu 24.04 cloud images often ship a drop-in that overrides sshd_config --
# files under sshd_config.d/ are read after the main file and win on conflicts.
# Check for one before assuming an edit to sshd_config alone takes effect:
grep -rl "PasswordAuthentication" /etc/ssh/sshd_config.d/ 2>/dev/null

sudo sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
# if the grep above found a match, edit that file the same way instead --
# it silently wins over the change above otherwise

sudo sshd -T | grep -i passwordauthentication   # must print "passwordauthentication no"
```

Before restarting `sshd`: open a **second**, separate SSH session and confirm
you can still log in as `rtrec` with your key while the first session stays
connected. Only once that second login succeeds:

```bash
sudo systemctl restart sshd
```

If you skip the second-session check and key login is somehow broken, this
is how a host gets permanently locked out — restarting `sshd` with only one
already-open, unaffected session is not a safety net.

## 4. Optional: reverse proxy and TLS for a public demo

Everything in §1-§3 stands on its own — the stack runs, and you can reach
`/recommend` and every ops/showcase service over the SSH tunnel from §3
without any of this. This section is only needed if you want `/recommend`
reachable from the public internet without a tunnel (e.g. to share the demo
link in the README). It was skipped entirely on the reference deploy this
doc is based on, and nothing broke — the stack is fully usable without it.

If you do want it: only `/recommend` (via `api`) needs to be public. Grafana
and the Streamlit dashboard default to **SSH-tunnel-only** access — they're
ops/showcase tools for you, not the public surface, and Grafana's
`GF_AUTH_ANONYMOUS_ENABLED=true` means anyone who could reach it would have
full viewer access with no login. If you want them public anyway, add HTTP
basic auth (shown below) rather than relying on anonymous mode.

Install nginx + certbot on the host (not in compose, since it needs port 80
before TLS exists):

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
```

`/etc/nginx/sites-available/rtrec`:

```nginx
limit_req_zone $binary_remote_addr zone=recommend:10m rate=5r/s;

server {
    listen 80;
    server_name <your-host>;

    location /recommend {
        limit_req zone=recommend burst=20 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /health {
        proxy_pass http://127.0.0.1:8000;
    }

    # Uncomment to expose Grafana/Streamlit publicly behind basic auth
    # instead of tunnel-only access (see below for htpasswd setup):
    # location /grafana/ {
    #     auth_basic "restricted";
    #     auth_basic_user_file /etc/nginx/.htpasswd;
    #     proxy_pass http://127.0.0.1:3000/;
    # }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/rtrec /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d <your-host>   # rewrites the server block for TLS,
                                       # installs a systemd timer for renewal
```

5 req/s with a burst of 20 per client IP on `/recommend` is a starting point
for a demo, not a tuned production limit — adjust to what the showcase
actually needs.

**Default (recommended): Grafana/Streamlit stay off the public internet** —
reached via the SSH tunnel in [§3](#3-firewall-and-ssh-hardening-required-for-any-host-left-running).
**If you want them public instead**, add basic auth:

```bash
sudo apt-get install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd <username>
# uncomment the /grafana/ location block above, add an equivalent for /streamlit/,
# and set GF_SERVER_ROOT_URL / GF_SERVER_SERVE_FROM_SUB_PATH in docker-compose.yml
# if serving Grafana under a subpath
```

## 5. Disk: the simulator never stops

The simulator produces continuously (`EVENTS_PER_SECOND=8` by default) — this
is a running service, not a fixed dataset. Two things bound disk growth, and
both are already wired, just need production-appropriate values:

- **Postgres**: `services/sink/app/main.py` already runs
  `drop_old_event_partitions(SINK_RETENTION_DAYS)` on an hourly timer — no
  cron needed. At this project's write rate, `pg_database_size` measured
  ~1.7GB for 4.76M events (~360 bytes/event including indexes). At 8
  events/s that's ~250MB/day; 30 days (the dev default) is ~7.5GB, which
  fits `vc2-4c-8gb`'s 160GB disk without trouble, but 7 days (~1.75GB) is
  the recommended production value for headroom and is set in [§2](#2-clone-and-configure).
- **Redpanda**: topic retention defaults to the cluster setting
  (`log_retention_ms`, 7 days out of the box) with no `retention_bytes` cap.
  Measured ~712MB of broker data for the same window. Leave the default
  unless you lower `EVENTS_PER_SECOND` or extend retention — if you do,
  cap it explicitly: `rpk topic alter-config user_events --set retention.bytes=5368709120` (5GB).

Check actual usage periodically: `df -h`, `docker system df`, or
`docker exec rtrec-postgres psql -U rtrec -d rtrec -c "SELECT pg_size_pretty(pg_database_size('rtrec'));"`.

## 6. Verify

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
# ranker_loaded and two_tower_loaded should both be true, warnings: []

curl -s "http://127.0.0.1:8000/recommend?user_id=u_42&k=5" | python3 -m json.tool
# items[].reason should be "lightgbm_ranker" / "retrieved_via": "two_tower"
# for a warm user id, not a heuristic/popularity fallback
```

(Swap `http://127.0.0.1:8000` for `https://<your-host>` if you did §4 and want
to check the public path specifically.)

### Latency: sweep concurrency, don't trust a single measurement

A single request at c=1 measures nothing about throughput under load, and
this stack's real bottleneck (§1's `UVICORN_WORKERS`) only shows up as a
**concurrency** effect: with 1 worker, throughput stays flat across c=1/4/10
instead of scaling, because synchronous LightGBM/PyTorch inference blocks
uvicorn's single event loop. Sweep concurrency and record throughput
alongside latency, not latency alone — flat req/s across increasing
concurrency is the signature to look for, and it's invisible in a c=1 test.

The README's latency table is a **service** number: measured directly
against `127.0.0.1:8000`, no TLS, no nginx, no network hop. If you also did
§4, measure the public URL separately and label it explicitly as end-to-end
(it will include network RTT, TLS handshake, and nginx — hit from far
outside the region you picked, that's easily 100-200ms+ of unrelated travel
time on top, and reads as a regression that never happened if conflated with
the service number). If you didn't do §4, don't leave a placeholder for it —
report only what was actually measured.

```bash
# one-off static binary, no build toolchain needed:
curl -sL https://github.com/hatoo/oha/releases/latest/download/oha-linux-amd64 -o oha
chmod +x oha && sudo mv oha /usr/local/bin/

for c in 1 4 10; do
  echo "=== concurrency $c ==="
  oha -z 30s -c "$c" "http://127.0.0.1:8000/recommend?user_id=u_42&k=10"
done
# reads p50/p90/p99 and req/s directly off oha's output; repeat with
# UVICORN_WORKERS set to the host's vCPU count and compare
```

Zero-dependency fallback if you'd rather not install anything (single
concurrency level only — no throughput signal, so it will miss the
serialization bug entirely; use `oha` if you can):

```bash
for i in $(seq 50); do curl -o /dev/null -s -w "%{time_total}\n" \
  "http://127.0.0.1:8000/recommend?user_id=u_$((RANDOM % 5000))&k=10"; done | sort -n
# p50 = the 25th value, p95 = the 48th, out of 50 sorted samples
```

Report the service number in the README's latency table (workers x
concurrency x p50/p95/p99/throughput). Only add an end-to-end row if §4 was
actually done — label it explicitly ("end-to-end, from `<your location>`,
includes network + TLS + nginx)".

### Everything else still running

Over the SSH tunnel: Grafana (`:3000`) shows live panels, and `make rows` /
the Streamlit dashboard's Event Stream tab confirm the simulator and sink
are both still producing on the new host.

## 7. Tear down, and restoring from a snapshot

A running instance only earns its cost while something's actively being
measured or demoed against it — once deploy process and latency numbers are
validated, there's nothing a live host is doing that a snapshot doesn't
preserve for a fraction of the price (cents/month for the snapshot vs. the
full instance rate). Snapshot before deleting the instance.

Restoring brings back the fully provisioned host as it was at snapshot
time — Compose stack, `ufw` rules, SSH key auth all intact — in a few
minutes. The one thing that changes is the IP:

```bash
ssh -L 3000:localhost:3000 -L 8501:localhost:8501 rtrec@<new-ip>
# add -L 8080:localhost:8080 -L 9090:localhost:9090 too if you need
# Redpanda Console / Prometheus, per §3
```

`docker compose ps` after reconnecting to confirm all containers came back
healthy — a restored instance is still a boot event for everything in the
stack, same as a reboot of the original host would be. Redis has no
persistence by design (§4/§5 elsewhere in this project), so online-store
state is gone regardless of snapshot/restore and refills from the live
stream within seconds, same as any other restart. Postgres data is part of
the snapshot and comes back intact.
