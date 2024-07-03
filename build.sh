# build wheel
python setup.py bdist_wheel

RED='\033[0;31m'
NC='\033[0m' 
if [ $? -eq 0 ]; then
    echo -e "${RED} ERROR building wheel!!!${NC}"
    exit 1
fi

# install to site-packages
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Installing on Mac"
    pip install dist/tokenizers_cpp-0.0.1-cp310-cp310-macosx_11_0_arm64.whl --force-reinstall
else
    # assume line
    echo "Assuming Linux. Installing on Linux"
    pip install dist/tokenizers_cpp-0.0.1-cp310-cp310-linux_x86_64.whl --force-reinstall
fi
