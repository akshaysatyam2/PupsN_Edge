#!/bin/bash
set -e

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing YOLO dependencies..."
pip install --quiet ultralytics onnx

echo "Running YOLO Export..."
python export_yolo.py

echo "Installing OSNet dependencies..."
pip install --quiet huggingface_hub torch torchvision yacs scipy gdown setuptools

if [ ! -d "deep-person-reid" ]; then
    echo "Cloning deep-person-reid repository..."
    git clone https://github.com/KaiyangZhou/deep-person-reid.git
fi

cd deep-person-reid
echo "Installing deep-person-reid requirements..."
pip install -r requirements.txt --quiet
echo "Setting up deep-person-reid..."
python setup.py develop --quiet
cd ..

echo "Running OSNet Export..."
python export_osnet.py

echo "Done!"
