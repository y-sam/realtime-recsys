# Deploying to a public host

Target: a single ARM64 VPS running `docker compose` — Hetzner **CAX21** (4 vCPU,
8GB RAM, 80GB NVMe, ~€6.60/mo) or Oracle Cloud **Always Free Ampere A1**
(up to 4 OCPU / 24GB RAM / 200GB block storage, $0/mo but capacity in the free
tier is not always available in every region). Either works; CAX21 is the
easier one to actually get provisioned.

ARM64 was chosen deliberately, not defaulted into: every image in
`docker-compose.yml` (`redpandadata/redpanda`, `redpandadata/console`,
`redis:7.4-alpine`, `postgres:16-alpine`, `prom/prometheus`,
`grafana/grafana`) publishes an official `linux/arm64` manifest, and the five
custom images (`simulator`, `consumer`, `sink`, `api`, `dashboard`) all build
from `python:3.12-slim`, itself multi-arch. This isn't a guess: the whole
stack — including `api`, which links LightGBM and needs `libgomp1` — has been
built and run natively on `linux/arm64` throughout this project's local
development. No image changes needed.

## 1. Provision

- Ubuntu 24.04 LTS, ARM64 image.
- Add your SSH key at creation; disable password auth.
- Minimum disk: 40GB. See [§5](#5-disk-the-simulator-never-stops) before
  picking retention settings — the simulator produces continuously, so disk
  is the one resource that grows unbounded if left on defaults.

```bash
ssh root@<host>
adduser rtrec && usermod -aG sudo rtrec
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
  not `ports:`, for it — see [§3](#3-firewall)), so the password is defense
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

## 3. Firewall — defense in depth, not the primary control

The primary control already lives in `docker-compose.yml`: `postgres` and
`redis` use `expose:` (compose-internal network only, no host port at all),
and `api`, `redpanda`, `console`, `prometheus`, and `grafana`/`dashboard`
(the latter two only start with the `dev` profile) publish to
`127.0.0.1:<port>:<port>`, not `0.0.0.0`. None of those ports are reachable
from the internet regardless of the firewall — that was the actual fix for
"a service got exposed that shouldn't have been." `ufw` below is a second
layer in case something is ever added to compose without the same care, not
the thing keeping Postgres off the internet today.

```bash
sudo ufw default deny incoming
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

Only 80 and 443 (nginx, [§4](#4-reverse-proxy-tls--rate-limiting)) need to be
open to the internet. Everything else is reached over an SSH tunnel:

```bash
ssh -L 3000:localhost:3000 -L 8501:localhost:8501 -L 8080:localhost:8080 \
    -L 9090:localhost:9090 rtrec@<host>
# then open http://localhost:<port> locally for Grafana / Streamlit / Redpanda Console / Prometheus
```

## 4. Reverse proxy: TLS + rate limiting

Only `/recommend` (via `api`) needs to be public for the demo link in the
README. Grafana and the Streamlit dashboard default to **SSH-tunnel-only**
access — they're ops/showcase tools for you, not the public surface, and
Grafana's `GF_AUTH_ANONYMOUS_ENABLED=true` means anyone who could reach it
would have full viewer access with no login. If you want them public anyway,
add HTTP basic auth (shown below) rather than relying on anonymous mode.

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
reached via the SSH tunnel in [§3](#3-firewall---defense-in-depth-not-the-primary-control).
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
  fits a CAX21's 80GB disk without trouble, but 7 days (~1.75GB) is the
  recommended production value for headroom and is set in [§2](#2-clone-and-configure).
- **Redpanda**: topic retention defaults to the cluster setting
  (`log_retention_ms`, 7 days out of the box) with no `retention_bytes` cap.
  Measured ~712MB of broker data for the same window. Leave the default
  unless you lower `EVENTS_PER_SECOND` or extend retention — if you do,
  cap it explicitly: `rpk topic alter-config user_events --set retention.bytes=5368709120` (5GB).

Check actual usage periodically: `df -h`, `docker system df`, or
`docker exec rtrec-postgres psql -U rtrec -d rtrec -c "SELECT pg_size_pretty(pg_database_size('rtrec'));"`.

## 6. Verify

```bash
curl -s https://<your-host>/health | python3 -m json.tool
# ranker_loaded and two_tower_loaded should both be true, warnings: []

curl -s "https://<your-host>/recommend?user_id=u_42&k=5" | python3 -m json.tool
# items[].reason should be "lightgbm_ranker" / "retrieved_via": "two_tower"
# for a warm user id, not a heuristic/popularity fallback
```

### Two different latency numbers — don't conflate them

The README's `p50 < 100ms` target is a **service** number: Redis I/O plus
model scoring, nothing else. Measuring it through `https://<host>/...` adds
network RTT, TLS handshake, and nginx — for a host in Germany hit from
Brazil that's 200ms+ of unrelated travel time stacked on top, and it would
read as a regression that never happened. Measure both, and keep them
labeled separately — don't put the end-to-end number where the README
claims the service number.

**(a) Service latency — on the host itself, no TLS, no nginx, the number
that goes in the README:**

```bash
# one-off static binary, no build toolchain needed on ARM64:
curl -sL https://github.com/hatoo/oha/releases/latest/download/oha-linux-arm64 -o oha
chmod +x oha && sudo mv oha /usr/local/bin/

oha -z 30s -c 10 "http://127.0.0.1:8000/recommend?user_id=u_42&k=10"
# reads p50/p90/p99 directly off oha's output — that's the number for the README
```

Zero-dependency fallback if you'd rather not install anything:

```bash
for i in $(seq 50); do curl -o /dev/null -s -w "%{time_total}\n" \
  "http://127.0.0.1:8000/recommend?user_id=u_$((RANDOM % 5000))&k=10"; done | sort -n
# p50 = the 25th value, p95 = the 48th, out of 50 sorted samples
```

**(b) End-to-end latency — from outside, through DNS/TLS/nginx, labeled as
what it actually measures:**

```bash
oha -z 30s -c 10 "https://<your-host>/recommend?user_id=u_42&k=10"
```

Report both in the README as separate fields (see the two placeholders
there) — "service p50/p95" and "end-to-end p50/p95 (from `<your location>`,
includes network + TLS + nginx)".

### Everything else still running

Over the SSH tunnel: Grafana (`:3000`) shows live panels, and `make rows` /
the Streamlit dashboard's Event Stream tab confirm the simulator and sink
are both still producing on the new host.
