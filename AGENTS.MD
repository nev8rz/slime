# Agent Notes

## Remote Debug Machine

Machine-specific debug connection details live in `LOCAL.MD`.

`LOCAL.MD` is intentionally gitignored and may contain environment variables, private host details, and local-only commands. Do not push it.

## W&B / proxy env vars

Some clusters can only reach `api.wandb.ai` through an HTTP proxy, whose value
lives in `<SLIME_ROOT>/.env` (e.g. `HTTP_PROXY`, `WANDB_HTTP(S)_PROXY`). Ray
worker processes don't always inherit a shell that sourced `.env`. Separately,
newer `wandb` validates proxy vars as URLs via pydantic and crashes on empty
strings ("Input should be a valid URL, input is empty").

Both are handled centrally in `slime/utils/wandb_utils.py`
(`_setup_proxy_env_vars`), called at the top of both wandb init entry points: it
loads proxy vars from `.env` (or `$SLIME_ENV_FILE`) when missing, drops empty
ones, and backfills `WANDB_HTTP(S)_PROXY`. **Do NOT re-add proxy bookkeeping to
launch/stage scripts** (e.g. the `RUNTIME_ENV_JSON` in
`examples/retool_post_train_all/*.sh`); keep those scripts free of proxy vars and
let `wandb_utils` handle it.
