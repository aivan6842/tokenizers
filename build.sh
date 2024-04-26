# build wheel
python setup.py bdist_wheel

# install to site-packages
pip install dist/tokenizers_cpp-0.0.1-cp310-cp310-linux_x86_64.whl --force-reinstall