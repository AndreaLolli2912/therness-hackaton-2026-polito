#!/bin/bash
# install_conda.sh: Installs Miniconda on Linux or macOS.

set -e

# Detect OS
OS_NAME=$(uname -s)
if [ "$OS_NAME" == "Linux" ]; then
    CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
elif [ "$OS_NAME" == "Darwin" ]; then
    ARCH_NAME=$(uname -m)
    if [ "$ARCH_NAME" == "arm64" ]; then
        CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    else
        CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    fi
else
    echo "Unsupported OS: $OS_NAME"
    exit 1
fi

INSTALLER="miniconda_installer.sh"
INSTALL_PATH="$HOME/miniconda3"

echo "Downloading Miniconda from $CONDA_URL..."
curl -L "$CONDA_URL" -o "$INSTALLER"

echo "Installing Miniconda to $INSTALL_PATH..."
bash "$INSTALLER" -b -p "$INSTALL_PATH"

# Cleanup
rm "$INSTALLER"

# Initialize conda for the current shell
source "$INSTALL_PATH/bin/activate"
conda init

echo "Miniconda installed successfully."
echo "Please restart your terminal or run 'source ~/.bashrc' (or ~/.zshrc) to use 'conda'."
