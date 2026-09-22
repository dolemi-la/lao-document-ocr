# Deployment presets

These files are examples, not secret-bearing production configuration.

The default `docker-compose.yml` remains the local-first base.

Available presets:

- `presets/local.env.example` — normal local/self-hosted use
- `presets/public-single-node.env.example` — bind services to localhost for a reverse proxy and enable a modest submission rate limit
- `presets/s3.env.example` — S3-compatible result storage
- `presets/gpu.env.example` — NVIDIA owned-recognizer runtime
- `compose.public.yml` — extra container hardening/resource limits for a single public node
- `compose.s3.yml` — installs the optional S3 dependencies and selects the S3 storage backend
- `compose.gpu.yml` — owned-recognizer API image, model mount, and GPU reservation

See `docs/deployment-presets.md` for commands and caveats.
