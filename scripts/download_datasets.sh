#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="./data"
MAX_REPAIR_ATTEMPTS=10
SKIP_IRIDIA_UNZIP=0
DATASET="all"

SHDB_URL="https://physionet.org/files/shdb-af/1.0.1"
SHDB_S3="s3://physionet-open/shdb-af/1.0.1"

CPSC_URL="https://physionet.org/files/cpsc2021/1.0.0"
CPSC_S3="s3://physionet-open/cpsc2021/1.0.0"

IRIDIA_URL="https://zenodo.org/records/8405941/files"
IRIDIA_METADATA="iridia-af-metadata-v1.0.1.csv"
IRIDIA_RECORDS="iridia-af-records-v1.0.1.zip"
IRIDIA_METADATA_MD5="f0102de28265ed6459956ac6cd397464"
IRIDIA_RECORDS_MD5="81d58d21c9835a601cd9eb6c68bf9feb"

usage() {
  cat <<'USAGE'
Download TinyAF datasets with checksum verification and repair retries.

Usage:
  bash scripts/download_datasets.sh [all|shdb|cpsc|iridia] [options]

Dataset:
  all                  Download SHDB-AF, CPSC2021, and IRIDIA-AF.
  shdb                 Download SHDB-AF.
  cpsc                 Download CPSC2021.
  iridia               Download IRIDIA-AF.

Other options:
  --dir PATH           Destination data directory. Default: ./data
  --max-retries N      Number of checksum repair attempts. Default: 10
  --skip-iridia-unzip  Verify IRIDIA-AF zip, but do not extract it.
  -h, --help           Show this help.

PhysioNet downloads use aws s3 sync when aws is installed, otherwise recursive wget.
USAGE
}

die() {
  echo "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "$2"
}

count_lines() {
  wc -l < "$1" | tr -d ' '
}

parse_args() {
  local dataset_seen=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dir)
        [[ $# -ge 2 ]] || die "Missing value for --dir"
        DATA_DIR="$2"
        shift 2
        ;;
      --max-retries)
        [[ $# -ge 2 ]] || die "Missing value for --max-retries"
        MAX_REPAIR_ATTEMPTS="$2"
        shift 2
        ;;
      --skip-iridia-unzip|--no-unzip)
        SKIP_IRIDIA_UNZIP=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      all|shdb|cpsc|iridia)
        if [[ "$dataset_seen" -eq 1 ]]; then
          die "Choose only one dataset: all, shdb, cpsc, or iridia."
        fi
        DATASET="$1"
        dataset_seen=1
        shift
        ;;
      --all|--shdb|--cpsc|--iridia)
        die "Dataset flags are not supported. Use one positional dataset: all, shdb, cpsc, or iridia."
        ;;
      --*)
        die "Unknown option: $1"
        ;;
      *)
        die "Unknown dataset: $1"
        ;;
    esac
  done

  case "$MAX_REPAIR_ATTEMPTS" in
    ""|0|*[!0-9]*) die "--max-retries must be a positive integer." ;;
  esac
}

safe_relpath() {
  case "$1" in
    ""|/*|..|../*|*/..|*/../*) return 1 ;;
    *) return 0 ;;
  esac
}

wants_dataset() {
  [[ "$DATASET" == "all" || "$DATASET" == "$1" ]]
}

hash_file() {
  local algo="$1"
  local path="$2"

  case "$algo" in
    sha256)
      if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
      else
        need_cmd shasum "sha256sum or shasum is required."
        shasum -a 256 "$path" | awk '{print $1}'
      fi
      ;;
    md5)
      if command -v md5sum >/dev/null 2>&1; then
        md5sum "$path" | awk '{print $1}'
      elif command -v md5 >/dev/null 2>&1; then
        md5 -q "$path"
      else
        need_cmd openssl "md5sum, md5, or openssl is required."
        openssl dgst -md5 -r "$path" | awk '{print $1}'
      fi
      ;;
    *)
      die "Unsupported checksum algorithm: $algo"
      ;;
  esac
}

manifest_paths() {
  awk 'NF >= 2 { path = $2; sub(/^\*/, "", path); print path }' "$1"
}

write_missing() {
  local root="$1"
  local manifest="$2"
  local out="$3"
  local path

  : > "$out"

  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    safe_relpath "$path" || die "Unsafe path in manifest: $path"
    [[ -f "$root/$path" ]] || echo "$path" >> "$out"
  done < <(manifest_paths "$manifest")
}

verify_manifest() {
  local root="$1"
  local algo="$2"
  local manifest="$3"
  local bad="$4"
  local log="$5"
  local expected
  local path
  local actual

  : > "$bad"
  : > "$log"

  while read -r expected path _rest; do
    [[ -z "${expected:-}" || -z "${path:-}" ]] && continue
    path="${path#\*}"
    safe_relpath "$path" || die "Unsafe path in manifest: $path"

    if [[ ! -f "$root/$path" ]]; then
      echo "$path: missing" >> "$log"
      echo "$path" >> "$bad"
      continue
    fi

    expected="$(printf '%s' "$expected" | tr '[:upper:]' '[:lower:]')"
    actual="$(hash_file "$algo" "$root/$path")"

    if [[ "$actual" == "$expected" ]]; then
      echo "$path: OK" >> "$log"
    else
      echo "$path: FAILED" >> "$log"
      echo "$path" >> "$bad"
    fi
  done < "$manifest"

  [[ "$(count_lines "$bad")" -eq 0 ]]
}

delete_listed() {
  local root="$1"
  local list="$2"
  local path

  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    safe_relpath "$path" || die "Unsafe path in repair list: $path"
    rm -f "$root/$path"
  done < "$list"
}

repair_until_verified() {
  local label="$1"
  local root="$2"
  local algo="$3"
  local manifest="$4"
  local download_list_fn="$5"
  local missing
  local bad
  local log
  local attempt

  missing="$(mktemp)"
  bad="$(mktemp)"
  log="$(mktemp)"

  attempt=1
  while [[ "$attempt" -le "$MAX_REPAIR_ATTEMPTS" ]]; do
    echo
    echo "$label repair attempt $attempt of $MAX_REPAIR_ATTEMPTS"

    write_missing "$root" "$manifest" "$missing"
    if [[ "$(count_lines "$missing")" -gt 0 ]]; then
      "$download_list_fn" "$missing"
      attempt=$((attempt + 1))
      continue
    fi

    if verify_manifest "$root" "$algo" "$manifest" "$bad" "$log"; then
      echo "Done. $label checksums passed."
      rm -f "$missing" "$bad" "$log"
      return 0
    fi

    echo "Repairing $(count_lines "$bad") bad $label file(s)..."
    delete_listed "$root" "$bad"
    "$download_list_fn" "$bad"
    attempt=$((attempt + 1))
  done

  echo "$label did not verify after $MAX_REPAIR_ATTEMPTS attempts." >&2
  echo "Last checksum output:" >&2
  cat "$log" >&2
  rm -f "$missing" "$bad" "$log"
  return 1
}

download_physionet_file() {
  local root="$1"
  local https_base="$2"
  local s3_base="$3"
  local path="$4"
  local dest="$root/$path"

  safe_relpath "$path" || die "Unsafe download path: $path"
  mkdir -p "$(dirname "$dest")"

  if command -v aws >/dev/null 2>&1; then
    aws s3 cp --no-sign-request "$s3_base/$path" "$dest"
  else
    need_cmd wget "wget is required when aws is not available."
    wget -c -O "$dest" "$https_base/$path"
  fi
}

download_physionet_bulk() {
  local root="$1"
  local https_base="$2"
  local s3_base="$3"

  if command -v aws >/dev/null 2>&1; then
    aws s3 sync --no-sign-request "$s3_base/" "$root/"
  else
    need_cmd wget "wget is required when aws is not available."
    wget -r -N -c -np -nH --cut-dirs=3 -P "$root" "$https_base/"
  fi
}

download_current_physionet_list() {
  local list="$1"
  local path

  echo "Downloading $(count_lines "$list") $CURRENT_LABEL file(s)..."

  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    echo "Downloading: $path"
    download_physionet_file "$CURRENT_ROOT" "$CURRENT_URL" "$CURRENT_S3" "$path"
  done < "$list"
}

download_physionet_dataset() {
  CURRENT_LABEL="$1"
  CURRENT_ROOT="$2"
  CURRENT_URL="$3"
  CURRENT_S3="$4"

  mkdir -p "$CURRENT_ROOT"

  if command -v aws >/dev/null 2>&1; then
    echo "Using AWS S3 sync for $CURRENT_LABEL."
  else
    echo "aws not found; using recursive wget for $CURRENT_LABEL."
  fi

  download_physionet_bulk "$CURRENT_ROOT" "$CURRENT_URL" "$CURRENT_S3"
  [[ -f "$CURRENT_ROOT/SHA256SUMS.txt" ]] || die "Missing $CURRENT_LABEL checksum manifest after download."
  repair_until_verified "$CURRENT_LABEL" "$CURRENT_ROOT" sha256 "$CURRENT_ROOT/SHA256SUMS.txt" download_current_physionet_list
}

write_iridia_manifest() {
  mkdir -p "$IRIDIA_ROOT"
  {
    printf '%s  %s\n' "$IRIDIA_METADATA_MD5" "$IRIDIA_METADATA"
    printf '%s  %s\n' "$IRIDIA_RECORDS_MD5" "$IRIDIA_RECORDS"
  } > "$IRIDIA_ROOT/MD5SUMS.txt"
}

iridia_url() {
  case "$1" in
    "$IRIDIA_METADATA"|"$IRIDIA_RECORDS") printf '%s/%s?download=1\n' "$IRIDIA_URL" "$1" ;;
    *) die "Unknown IRIDIA-AF file: $1" ;;
  esac
}

download_iridia_list() {
  local list="$1"
  local path

  need_cmd wget "wget is required to download IRIDIA-AF from Zenodo."
  echo "Downloading $(count_lines "$list") IRIDIA-AF file(s)..."

  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    safe_relpath "$path" || die "Unsafe IRIDIA-AF path: $path"
    echo "Downloading: $path"
    wget -c -O "$IRIDIA_ROOT/$path" "$(iridia_url "$path")"
  done < "$list"
}

download_iridia_dataset() {
  IRIDIA_ROOT="$DATA_DIR/iridia-af"

  write_iridia_manifest
  repair_until_verified "IRIDIA-AF" "$IRIDIA_ROOT" md5 "$IRIDIA_ROOT/MD5SUMS.txt" download_iridia_list

  if [[ "$SKIP_IRIDIA_UNZIP" -eq 0 ]]; then
    need_cmd unzip "unzip is required to extract IRIDIA-AF."
    mkdir -p "$IRIDIA_ROOT/records"
    unzip -q -n "$IRIDIA_ROOT/$IRIDIA_RECORDS" -d "$IRIDIA_ROOT/records"
  fi
}

main() {
  parse_args "$@"
  mkdir -p "$DATA_DIR"

  wants_dataset shdb &&
    download_physionet_dataset "SHDB-AF" "$DATA_DIR/shdb-af" "$SHDB_URL" "$SHDB_S3"

  wants_dataset cpsc &&
    download_physionet_dataset "CPSC2021" "$DATA_DIR/cpsc_2021" "$CPSC_URL" "$CPSC_S3"

  wants_dataset iridia &&
    download_iridia_dataset

  echo
  echo "Done. Requested datasets downloaded and verified."
}

main "$@"
