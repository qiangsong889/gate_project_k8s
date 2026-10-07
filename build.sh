#!/usr/bin/env bash
# 用法：
#   ./build.sh v0.2        构建 linux/amd64 镜像并推送 $REG/kb-assistant:v0.2
# 可选环境变量：
#   REG        镜像仓库前缀，默认 docker.io/shyanetech
#   NOTES_SRC  笔记源目录，默认 k8s-notes-site 的 src/content
set -euo pipefail

version="${1:-}"
if [[ -z "$version" ]]; then
  echo "用法: ./build.sh <version>   例如 ./build.sh v0.2" >&2
  exit 1
fi
if [[ "$version" == "latest" ]]; then
  echo "不要用 latest，请给一个具体版本号" >&2
  exit 1
fi

REG="${REG:-docker.io/shyanetech}"
NOTES_SRC="${NOTES_SRC:-/Users/shaynesong/randomshit/k8s/k8s-notes-site/src/content}"
image="$REG/kb-assistant:$version"

if [[ ! -d "$NOTES_SRC" ]]; then
  echo "笔记目录不存在: $NOTES_SRC" >&2
  exit 1
fi

# 脚本放在项目根目录，无论从哪里调用都以它所在目录作为构建上下文
cd "$(dirname "$0")"

echo "==> 构建并推送 $image"
echo "    笔记来源: $NOTES_SRC"
docker buildx build --platform linux/amd64 \
  --build-context notes="$NOTES_SRC" \
  -t "$image" \
  --push .

echo "==> 完成：$image"
