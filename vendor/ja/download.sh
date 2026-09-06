#!/usr/bin/env bash

set -euo pipefail

VERSION="20260723"
BASE_URL="https://d2ej7fkh96fzlu.cloudfront.net/sudachidict-raw"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

download() {
	local url="$1"
	local output="$2"

	echo "Downloading $output..."
	curl -fL "$url" -o "$output"
}

download \
	"$BASE_URL/$VERSION/small_lex.zip" \
	"small_lex.zip"

download \
	"$BASE_URL/$VERSION/core_lex.zip" \
	"core_lex.zip"

download \
	"$BASE_URL/matrix.def.zip" \
	"matrix.def.zip"

echo "Extracting..."

unzip -o small_lex.zip
unzip -o core_lex.zip
unzip -o matrix.def.zip

rm -f \
	small_lex.zip \
	core_lex.zip \
	matrix.def.zip

echo "Done."