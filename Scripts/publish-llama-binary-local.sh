#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAG=""
PUBLISH=""
PRERELEASE=1
SOURCE_REF="HEAD"
REPOSITORY=""
REMOTE_NAME="origin"
SKIP_PUBLISHED_VALIDATION=0
RELEASE_NOTES=""
MANIFEST_BACKUP=""
TEMP_INDEX=""
RELEASE_COMMIT=""

usage() {
  cat <<'USAGE'
usage: Scripts/publish-llama-binary-local.sh --tag <vX.Y.Z> (--publish | --dry-run) [options]

Builds the llama XCFramework locally and optionally publishes the same
tag-only GitHub release that .github/workflows/publish-llama-binary.yml creates.

The script requires a clean current checkout, reuses its ignored build caches,
and restores Package.swift after validation. Publishing creates a tag-only
release commit without moving the current branch. Choose a mode explicitly.

Options:
  --tag <tag>                    Release tag to prepare, for example v0.3.0.
  --publish                      Commit, tag, push the tag, and create a GitHub release.
  --dry-run                      Build and validate without committing, tagging, or uploading.
  --prerelease                   Mark the GitHub release as a prerelease. Default.
  --no-prerelease                Publish as a stable GitHub release.
  --source-ref <ref>             Source commit/ref to release; it must equal HEAD. Default: HEAD.
  --remote <name>                Git remote to check and push tags to. Default: origin.
  --repo <owner/name>            GitHub repository. Default: parsed from origin.
  --skip-published-validation    Skip the post-upload clean consumer validation.
  -h, --help                     Show this help.
USAGE
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --tag)
      if [[ "$#" -lt 2 ]]; then
        echo "error: --tag requires a value" >&2
        exit 2
      fi
      TAG="${2:-}"
      shift 2
      ;;
    --publish)
      if [[ -n "$PUBLISH" ]]; then
        echo "error: release mode provided more than once" >&2
        exit 2
      fi
      PUBLISH=1
      shift
      ;;
    --dry-run)
      if [[ -n "$PUBLISH" ]]; then
        echo "error: release mode provided more than once" >&2
        exit 2
      fi
      PUBLISH=0
      shift
      ;;
    --prerelease)
      PRERELEASE=1
      shift
      ;;
    --no-prerelease)
      PRERELEASE=0
      shift
      ;;
    --source-ref)
      if [[ "$#" -lt 2 ]]; then
        echo "error: --source-ref requires a value" >&2
        exit 2
      fi
      SOURCE_REF="${2:-}"
      shift 2
      ;;
    --remote)
      if [[ "$#" -lt 2 ]]; then
        echo "error: --remote requires a value" >&2
        exit 2
      fi
      REMOTE_NAME="${2:-}"
      shift 2
      ;;
    --repo)
      if [[ "$#" -lt 2 ]]; then
        echo "error: --repo requires a value" >&2
        exit 2
      fi
      REPOSITORY="${2:-}"
      shift 2
      ;;
    --skip-published-validation)
      SKIP_PUBLISHED_VALIDATION=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    v[0-9]*|[0-9]*)
      if [[ -n "$TAG" ]]; then
        echo "error: tag provided more than once" >&2
        usage >&2
        exit 2
      fi
      TAG="$1"
      shift
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TAG" ]]; then
  echo "error: --tag is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "$PUBLISH" ]]; then
  echo "error: choose exactly one release mode: --publish or --dry-run" >&2
  usage >&2
  exit 2
fi

VERSION="${TAG#v}"
if [[ ! "$VERSION" =~ ^[0-9]+[.][0-9]+[.][0-9]+([.-][0-9A-Za-z.-]+)?([+][0-9A-Za-z.-]+)?$ ]]; then
  echo "error: release tag must be a SwiftPM semantic version tag, optionally prefixed with v: $TAG" >&2
  exit 1
fi

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "error: missing required command: $command_name" >&2
    exit 1
  fi
}

check_xcode() {
  local version_line=""
  local build_line=""
  local version=""
  local build_version=""

  while IFS= read -r line; do
    case "$line" in
      Xcode\ *) version_line="$line" ;;
      Build\ version\ *) build_line="$line" ;;
    esac
  done < <(xcodebuild -version)

  version="${version_line#Xcode }"
  build_version="${build_line#Build version }"

  if [[ -z "$version" || "$version" == "$version_line" ]]; then
    echo "error: unable to determine selected Xcode version" >&2
    exit 1
  fi

  if [[ ! "$version" =~ ^[0-9]+([.][0-9]+)*$ ]]; then
    echo "error: selected Xcode must be a stable release, found version: $version" >&2
    exit 1
  fi

  if [[ -z "$build_version" || "$build_version" == "$build_line" ]]; then
    echo "error: unable to determine selected Xcode build version" >&2
    exit 1
  fi

  case "$build_version" in
    *[A-Za-z])
      echo "error: selected Xcode build appears to be a prerelease seed: $build_version" >&2
      exit 1
      ;;
  esac

  local major="${version%%.*}"
  local rest="${version#*.}"
  local minor="0"
  if [[ "$rest" != "$version" ]]; then
    minor="${rest%%.*}"
  fi

  if [[ "$major" -lt 26 ]] || { [[ "$major" -eq 26 ]] && [[ "$minor" -lt 4 ]]; }; then
    echo "error: selected Xcode must be 26.4 or newer, found $version" >&2
    exit 1
  fi

  xcrun --sdk iphoneos --show-sdk-path >/dev/null
  xcrun --sdk iphonesimulator --show-sdk-path >/dev/null

  local sdk_list
  sdk_list="$(xcodebuild -showsdks 2>/dev/null || true)"
  if ! printf '%s\n' "$sdk_list" | grep -q -- '-sdk iphoneos'; then
    echo "error: selected Xcode does not report an iphoneos SDK" >&2
    exit 1
  fi
  if ! printf '%s\n' "$sdk_list" | grep -q -- '-sdk iphonesimulator'; then
    echo "error: selected Xcode does not report an iphonesimulator SDK" >&2
    exit 1
  fi
}

detect_repository() {
  local remote_url
  remote_url="$(git -C "$ROOT_DIR" remote get-url "$REMOTE_NAME" 2>/dev/null || true)"

  case "$remote_url" in
    https://github.com/*.git)
      remote_url="${remote_url#https://github.com/}"
      printf '%s\n' "${remote_url%.git}"
      ;;
    https://github.com/*)
      printf '%s\n' "${remote_url#https://github.com/}"
      ;;
    git@github.com:*.git)
      remote_url="${remote_url#git@github.com:}"
      printf '%s\n' "${remote_url%.git}"
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ -z "$REPOSITORY" ]]; then
  if ! REPOSITORY="$(detect_repository)"; then
    echo "error: could not infer GitHub repository from origin; pass --repo owner/name" >&2
    exit 1
  fi
fi

if [[ ! "$REPOSITORY" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  echo "error: repository must use owner/name form: $REPOSITORY" >&2
  exit 1
fi

require_command git
require_command swift
require_command xcodebuild
require_command xcrun
require_command cmake

check_xcode

if [[ "$PUBLISH" == "1" ]]; then
  require_command gh
  gh auth status --hostname github.com >/dev/null
  git -C "$ROOT_DIR" var GIT_AUTHOR_IDENT >/dev/null
  git -C "$ROOT_DIR" var GIT_COMMITTER_IDENT >/dev/null
fi

if [[ -z "$SOURCE_REF" ]]; then
  echo "error: --source-ref cannot be empty" >&2
  exit 2
fi

if [[ -z "$REMOTE_NAME" ]]; then
  echo "error: --remote cannot be empty" >&2
  exit 2
fi

if ! SOURCE_COMMIT="$(git -C "$ROOT_DIR" rev-parse --verify "$SOURCE_REF^{commit}" 2>/dev/null)"; then
  echo "error: source ref does not resolve to a commit: $SOURCE_REF" >&2
  exit 1
fi

HEAD_COMMIT="$(git -C "$ROOT_DIR" rev-parse --verify HEAD^{commit})"
if [[ "$SOURCE_COMMIT" != "$HEAD_COMMIT" ]]; then
  echo "error: --source-ref must resolve to the current HEAD; check it out before releasing" >&2
  echo "       this avoids creating a second checkout and duplicating its build cache" >&2
  exit 1
fi

if [[ -n "$(git -C "$ROOT_DIR" status --porcelain)" ]]; then
  echo "error: current checkout is dirty; commit or stash changes before releasing" >&2
  exit 1
fi

if git -C "$ROOT_DIR" rev-parse "$TAG" >/dev/null 2>&1; then
  echo "error: tag already exists locally: $TAG" >&2
  exit 1
fi

if [[ "$PUBLISH" == "1" ]]; then
  if git -C "$ROOT_DIR" ls-remote --exit-code --tags "$REMOTE_NAME" "refs/tags/$TAG" >/dev/null 2>&1; then
    echo "error: tag already exists on $REMOTE_NAME: $TAG" >&2
    exit 1
  fi
fi

restore_manifest() {
  if [[ -n "$MANIFEST_BACKUP" && -f "$MANIFEST_BACKUP" ]]; then
    cp "$MANIFEST_BACKUP" "$ROOT_DIR/Package.swift"
  fi
}

cleanup_release_staging() {
  local exit_status="$?"
  set +e

  restore_manifest
  if [[ -n "$MANIFEST_BACKUP" ]]; then
    rm -f "$MANIFEST_BACKUP"
  fi
  if [[ -n "$TEMP_INDEX" ]]; then
    rm -f "$TEMP_INDEX" "$TEMP_INDEX.lock"
  fi
  if [[ -n "$RELEASE_NOTES" ]]; then
    rm -f "$RELEASE_NOTES"
  fi

  return "$exit_status"
}

MANIFEST_BACKUP="$(mktemp "${TMPDIR:-/tmp}/cllm-package-swift.XXXXXX")"
cp "$ROOT_DIR/Package.swift" "$MANIFEST_BACKUP"
trap cleanup_release_staging EXIT

cd "$ROOT_DIR"
git submodule update --init --recursive

Scripts/build-llama-xcframework.sh

CHECKSUM_PATH="Vendor/llama-artifacts/release/llama.xcframework.zip.checksum"
ZIP_PATH="Vendor/llama-artifacts/release/llama.xcframework.zip"
CHECKSUM="$(cat "$CHECKSUM_PATH")"
ARTIFACT_URL="https://github.com/$REPOSITORY/releases/download/$TAG/llama.xcframework.zip"

Scripts/set-llama-binary-artifact.sh "$ARTIFACT_URL" "$CHECKSUM"
Scripts/validate-local-binary-artifact.sh

if [[ "$PUBLISH" == "0" ]]; then
  cat <<EOF
Dry run complete

Tag: $TAG
Source: $SOURCE_COMMIT
Asset: $ROOT_DIR/$ZIP_PATH
URL: $ARTIFACT_URL
Checksum: $CHECKSUM
Checkout: $ROOT_DIR (Package.swift restored on exit)
EOF
  exit 0
fi

TEMP_INDEX="$(mktemp "${TMPDIR:-/tmp}/cllm-release-index.XXXXXX")"
rm -f "$TEMP_INDEX"
GIT_INDEX_FILE="$TEMP_INDEX" git read-tree "$SOURCE_COMMIT^{tree}"
PACKAGE_BLOB="$(git hash-object -w Package.swift)"
GIT_INDEX_FILE="$TEMP_INDEX" git update-index \
  --add --cacheinfo "100644,$PACKAGE_BLOB,Package.swift"
GIT_INDEX_FILE="$TEMP_INDEX" git diff --cached --check "$SOURCE_COMMIT"
RELEASE_TREE="$(GIT_INDEX_FILE="$TEMP_INDEX" git write-tree)"
RELEASE_COMMIT="$(printf 'Publish llama binary artifact %s\n' "$TAG" \
  | git commit-tree "$RELEASE_TREE" -p "$SOURCE_COMMIT")"

CHANGED_PATHS="$(git diff-tree --no-commit-id --name-only -r "$SOURCE_COMMIT" "$RELEASE_COMMIT")"
if [[ "$CHANGED_PATHS" != "Package.swift" ]]; then
  echo "error: release commit changed unexpected paths:" >&2
  printf '%s\n' "$CHANGED_PATHS" >&2
  exit 1
fi

restore_manifest
if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: checkout changed during release validation; refusing to publish" >&2
  git status --short >&2
  exit 1
fi

git tag "$TAG" "$RELEASE_COMMIT"
git push "$REMOTE_NAME" "$TAG"

RELEASE_NOTES="$(mktemp "${TMPDIR:-/tmp}/cllm-release-notes.XXXXXX")"
cat > "$RELEASE_NOTES" <<EOF
SwiftPM binary artifact for CarbocationLocalLLM's llama runtime.

This archive redistributes static llama.cpp/ggml object code through
libllama-combined.a. See THIRD_PARTY_NOTICES.md in this release tag
for bundled and linked notices.

Checksum:
\`$CHECKSUM\`
EOF

release_args=(
  release
  create
  "$TAG"
  "$ZIP_PATH"
  --repo "$REPOSITORY"
  --title "$TAG"
  --notes-file "$RELEASE_NOTES"
)

if [[ "$PRERELEASE" == "1" ]]; then
  release_args+=(--prerelease)
fi

gh "${release_args[@]}"

if [[ "$SKIP_PUBLISHED_VALIDATION" == "0" ]]; then
  CARBOCATION_LOCAL_LLM_REPO_URL="https://github.com/$REPOSITORY.git" \
    Scripts/validate-published-binary-release.sh "$TAG"
fi

cat <<EOF
Release published

Tag: $TAG
Source: $SOURCE_COMMIT
Asset: $ROOT_DIR/$ZIP_PATH
URL: $ARTIFACT_URL
Checksum: $CHECKSUM
Release commit: $RELEASE_COMMIT
Checkout: $ROOT_DIR (branch unchanged)
EOF
