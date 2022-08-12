git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip3 install -r requirements/build.txt
pip3 install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip3 install -v -e .