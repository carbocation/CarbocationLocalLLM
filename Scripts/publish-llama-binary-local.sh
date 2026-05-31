#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_WORKTREE="$ROOT_DIR/.build/llama-release-worktree"

TAG=""
PUBLISH=0
PRERELEASE=1
SOURCE_REF="HEAD"
WORKTREE="$DEFAULT_WORKTREE"
REPOSITORY=""
REMOTE_NAME="origin"
KEEP_PACKAGE_SWIFT_DIRTY=0
SKIP_PUBLISHED_VALIDATION=0
RELEASE_NOTES=""

usage() {
  cat <<'USAGE'
usage: Scripts/publish-llama-binary-local.sh --tag <vX.Y.Z> [options]

Builds the llama XCFramework locally and optionally publishes the same
tag-only GitHub release that .github/workflows/publish-llama-binary.yml creates.

Defaults to a dry run. Dry runs build, stamp Package.swift in an isolated
worktree, validate against the local XCFramework, then restore Package.swift.

Options:
  --tag <tag>                    Release tag to prepare, for example v0.3.0.
  --publish                      Commit, tag, push the tag, and create a GitHub release.
  --dry-run                      Build and validate without committing, tagging, or uploading.
  --prerelease                   Mark the GitHub release as a prerelease. Default.
  --no-prerelease                Publish as a stable GitHub release.
  --source-ref <ref>             Source commit/ref to release. Default: HEAD.
  --worktree <path>              Reusable isolated worktree. Default: .build/llama-release-worktree.
  --remote <name>                Git remote to check and push tags to. Default: origin.
  --repo <owner/name>            GitHub repository. Default: parsed from origin.
  --skip-published-validation    Skip post-upload clean consumer/app validation.
  --keep-package-swift-dirty     Leave stamped Package.swift after a dry run.
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
      PUBLISH=1
      shift
      ;;
    --dry-run)
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
    --worktree)
      if [[ "$#" -lt 2 ]]; then
        echo "error: --worktree requires a value" >&2
        exit 2
      fi
      WORKTREE="${2:-}"
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
    --keep-package-swift-dirty)
      KEEP_PACKAGE_SWIFT_DIRTY=1
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
fi

if [[ -z "$SOURCE_REF" ]]; then
  echo "error: --source-ref cannot be empty" >&2
  exit 2
fi

if [[ -z "$WORKTREE" ]]; then
  echo "error: --worktree cannot be empty" >&2
  exit 2
fi

if [[ -z "$REMOTE_NAME" ]]; then
  echo "error: --remote cannot be empty" >&2
  exit 2
fi

case "$WORKTREE" in
  /*) ;;
  *) WORKTREE="$ROOT_DIR/$WORKTREE" ;;
esac

if ! SOURCE_COMMIT="$(git -C "$ROOT_DIR" rev-parse --verify "$SOURCE_REF^{commit}" 2>/dev/null)"; then
  echo "error: source ref does not resolve to a commit: $SOURCE_REF" >&2
  exit 1
fi

if [[ "$SOURCE_REF" == "HEAD" ]] && [[ -n "$(git -C "$ROOT_DIR" status --porcelain)" ]]; then
  echo "error: current worktree is dirty; commit/stash changes or pass --source-ref for an explicit clean ref" >&2
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

prepare_worktree() {
  local parent_dir
  parent_dir="$(dirname "$WORKTREE")"
  mkdir -p "$parent_dir"

  if [[ -e "$WORKTREE" ]]; then
    local worktree_root
    worktree_root="$(git -C "$WORKTREE" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ "$worktree_root" != "$WORKTREE" ]]; then
      echo "error: worktree path exists but is not a git worktree rooted there: $WORKTREE" >&2
      exit 1
    fi
    if [[ -n "$(git -C "$WORKTREE" status --porcelain)" ]]; then
      echo "error: release worktree is dirty: $WORKTREE" >&2
      echo "       clean it, remove it with 'git worktree remove', or pass --worktree for another path." >&2
      exit 1
    fi
    git -C "$WORKTREE" checkout --detach "$SOURCE_COMMIT"
  else
    git -C "$ROOT_DIR" worktree add --detach "$WORKTREE" "$SOURCE_COMMIT"
  fi
}

prepare_worktree

cd "$WORKTREE"
git submodule update --init --recursive

cleanup_dry_run_package_swift() {
  if [[ "$PUBLISH" == "0" && "$KEEP_PACKAGE_SWIFT_DIRTY" == "0" ]]; then
    git checkout -- Package.swift >/dev/null 2>&1 || true
  fi
  if [[ -n "$RELEASE_NOTES" ]]; then
    rm -f "$RELEASE_NOTES"
  fi
}

trap cleanup_dry_run_package_swift EXIT

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
Asset: $WORKTREE/$ZIP_PATH
URL: $ARTIFACT_URL
Checksum: $CHECKSUM
Worktree: $WORKTREE
EOF
  exit 0
fi

git add Package.swift
git diff --cached --check
git commit -m "Publish llama binary artifact $TAG"
git tag "$TAG"
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
Asset: $WORKTREE/$ZIP_PATH
URL: $ARTIFACT_URL
Checksum: $CHECKSUM
Worktree: $WORKTREE
EOF
