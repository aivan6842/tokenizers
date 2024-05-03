# build wheel
python setup.py bdist_wheel

# install to site-packages
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Installing on Mac"
    pip install dist/tokenizers_cpp-0.0.1-cp310-cp310-macosx_11_0_arm64.whl --force-reinstall
else
    # assume line
    echo "Assuming Linux. Installing on Linux"
    pip install dist/tokenizers_cpp-0.0.1-cp310-cp310-linux_x86_64.whl --force-reinstall
fi

