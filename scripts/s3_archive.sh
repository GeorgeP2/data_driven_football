#!/usr/bin/env bash
# Archive the outputs that cannot be regenerated to a personal S3 bucket, or pull them back.
#
#   scripts/s3_archive.sh sync      # local -> S3
#   scripts/s3_archive.sh restore   # S3 -> local
#
# Reads S3_BUCKET, AWS_PROFILE and AWS_ACCOUNT_ID from .env (see .env.example). This machine's
# [default] profile is a work account, so the profile is never defaulted: it must be named, must
# not be "default", and must resolve to the expected account before anything is copied.
# Neither direction passes --delete — the archive only grows, and a restore never removes local work.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

mode="${1:-}"
if [[ "$mode" != "sync" && "$mode" != "restore" ]]; then
  echo "usage: $0 sync|restore" >&2
  exit 2
fi

for var in S3_BUCKET AWS_PROFILE AWS_ACCOUNT_ID; do
  if [[ -z "${!var:-}" ]]; then
    echo "s3_archive: $var is not set. Copy .env.example to .env and fill it in." >&2
    exit 1
  fi
done

if [[ "$AWS_PROFILE" == "default" ]]; then
  echo "s3_archive: refusing AWS_PROFILE=default — that is not the personal account." >&2
  exit 1
fi

account="$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text)"
if [[ "$account" != "$AWS_ACCOUNT_ID" ]]; then
  echo "s3_archive: profile '$AWS_PROFILE' is account $account, expected $AWS_ACCOUNT_ID. Nothing copied." >&2
  exit 1
fi

# Paths that cannot be regenerated: pool forecasts and API snapshots are lost once the deadline
# passes, and FPL wipes entry and league history at the summer reset.
paths=(
  fpl_optimiser_project/outputs/forecast_log
  fpl_optimiser_project/outputs/snapshots
  fpl_optimiser_project/data
)

for path in "${paths[@]}"; do
  remote="s3://$S3_BUCKET/$path"
  if [[ "$mode" == "sync" ]]; then
    [[ -d "$path" ]] || continue
    aws s3 sync "$path" "$remote" --profile "$AWS_PROFILE" --exclude ".gitkeep" --exclude "README.md"
  else
    mkdir -p "$path"
    aws s3 sync "$remote" "$path" --profile "$AWS_PROFILE"
  fi
done
