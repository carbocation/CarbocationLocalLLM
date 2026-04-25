#!/usr/bin/env bash

set -euo pipefail

PACKAGE_NAME="${CARBOCATION_LOCAL_LLM_PACKAGE_NAME:-CarbocationLocalLLM}"
PACKAGE_NAME_LOWERCASE="$(printf '%s' "$PACKAGE_NAME" | tr '[:upper:]' '[:lower:]')"
BUILD_SCRIPT_RELATIVE_PATH="Scripts/build-llama-macos.sh"

candidate_roots=()
source_package_dirs=()

append_unique() {
  local value="$1"
  local existing

  if [[ -z "$value" ]]; then
    return
  fi

  if (( ${#candidate_roots[@]} > 0 )); then
    for existing in "${candidate_roots[@]}"; do
      if [[ "$existing" == "$value" ]]; then
        return
      fi
    done
  fi

  candidate_roots+=("$value")
}

append_source_packages_dir() {
  local value="$1"
  local existing

  if [[ -z "$value" || ! -d "$value" ]]; then
    return
  fi

  if (( ${#source_package_dirs[@]} > 0 )); then
    for existing in "${source_package_dirs[@]}"; do
      if [[ "$existing" == "$value" ]]; then
        return
      fi
    done
  fi

  source_package_dirs+=("$value")
}

canonical_dir() {
  local path="$1"

  if [[ -d "$path" ]]; then
    (cd "$path" && pwd -P)
  else
    return 1
  fi
}

script_dir() {
  local source="${BASH_SOURCE[0]}"
  local dir

  while [[ -L "$source" ]]; do
    dir="$(cd -P "$(dirname "$source")" && pwd)"
    source="$(readlink "$source")"
    if [[ "$source" != /* ]]; then
      source="$dir/$source"
    fi
  done

  cd -P "$(dirname "$source")" && pwd
}

add_candidate_root() {
  local raw="$1"
  local resolved

  if resolved="$(canonical_dir "$raw" 2>/dev/null)"; then
    append_unique "$resolved"
  fi
}

add_source_packages_from_path() {
  local raw="$1"
  local path

  if [[ -z "$raw" ]]; then
    return
  fi

  if [[ -d "$raw" ]]; then
    path="$(canonical_dir "$raw" 2>/dev/null || true)"
  else
    path="$(canonical_dir "$(dirname "$raw")" 2>/dev/null || true)"
  fi

  while [[ -n "$path" && "$path" != "/" ]]; do
    append_source_packages_dir "$path/SourcePackages/checkouts"
    append_source_packages_dir "$path/SourcePackages/repositories"
    path="$(dirname "$path")"
  done
}

if [[ -n "${CARBOCATION_LOCAL_LLM_ROOT:-}" ]]; then
  add_candidate_root "$CARBOCATION_LOCAL_LLM_ROOT"
fi

own_script_dir="$(script_dir)"
add_candidate_root "$own_script_dir/.."

if [[ -n "${SRCROOT:-}" ]]; then
  add_candidate_root "$SRCROOT/../$PACKAGE_NAME"
  add_source_packages_from_path "$SRCROOT"
fi

for build_path in \
  "${BUILD_DIR:-}" \
  "${BUILT_PRODUCTS_DIR:-}" \
  "${CONFIGURATION_BUILD_DIR:-}" \
  "${OBJROOT:-}" \
  "${PROJECT_TEMP_DIR:-}" \
  "${SYMROOT:-}"; do
  add_source_packages_from_path "$build_path"
done

if (( ${#source_package_dirs[@]} > 0 )); then
  for source_packages_dir in "${source_package_dirs[@]}"; do
    add_candidate_root "$source_packages_dir/$PACKAGE_NAME"
    add_candidate_root "$source_packages_dir/$PACKAGE_NAME_LOWERCASE"

    while IFS= read -r found_script; do
      add_candidate_root "$(dirname "$(dirname "$found_script")")"
    done < <(find "$source_packages_dir" -maxdepth 3 -path "*/$BUILD_SCRIPT_RELATIVE_PATH" -type f 2>/dev/null)
  done
fi

if (( ${#candidate_roots[@]} > 0 )); then
  for root in "${candidate_roots[@]}"; do
    build_script="$root/$BUILD_SCRIPT_RELATIVE_PATH"
    if [[ -x "$build_script" && -f "$root/Package.swift" ]]; then
      exec "$build_script"
    fi
  done
fi

cat >&2 <<EOF
error: could not locate $PACKAGE_NAME/$BUILD_SCRIPT_RELATIVE_PATH.

Set CARBOCATION_LOCAL_LLM_ROOT to the package checkout path for local development,
or make sure Xcode has resolved the Swift package dependency before this Run Script
phase executes.
EOF
exit 1
