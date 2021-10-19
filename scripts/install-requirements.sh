apt-get update
apt-get upgrade -y

pip install --upgrade pip setuptools wheel
pip install notebook

apt-get install git -y

pip install https://github.com/CellProfiling/HPA-Cell-Segmentation/archive/master.zip
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install pandas

apt-get install libgl1-mesa-glx -y
apt-get install libglib2.0-0 -y

pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension

pip install opencv-python

pip install efficientnet_pytorch
pip install ttach
pip install tqdm
pip install grad-cam

pip install openpyxl
