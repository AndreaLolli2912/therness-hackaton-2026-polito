#!/usr/bin/env bash

set -e  # stop on error

echo "📦 Downloading sampleData from GitHub..."

# Config (change if needed)
REPO_URL="https://github.com/Therness-Hackathon/ProjectDescription.git"
TARGET_DIR="$PWD/sampleData"
TMP_DIR="$PWD/tmp_projectdescription"

# Clean previous temp if exists
rm -rf "$TMP_DIR"

# Clone only sampleData using sparse checkout
git clone --filter=blob:none --sparse "$REPO_URL" "$TMP_DIR"

cd "$TMP_DIR"
git sparse-checkout set sampleData

echo "📂 Moving sampleData to working directory..."

# Remove old sampleData if exists
rm -rf "$TARGET_DIR"

mv sampleData "$TARGET_DIR"

cd ..
rm -rf "$TMP_DIR"

echo "✅ Done. sampleData is here:"
echo "$TARGET_DIR"

echo "📏 Size:"
du -sh "$TARGET_DIR"