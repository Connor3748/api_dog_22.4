
import subprocess
import os
subprocess.call('pip3 install -r api/requirements.txt', shell=True)
import torch
try:
  torch_version, cuda_version = torch.__version__.split('+')
except ValueError:
  torch_version, cuda_version = torch.__version__, subprocess.call('nvcc -V', shell=True)

file_name = 'mmstart.sh'
with open(file_name, 'w') as f:
    f.write(
        f'pip3 install mmcv-full==1.4.7 -f https://download.openmmlab.com/mmcv/dist/{cuda_version}/torch{torch_version}/index.html')
subprocess.call('bash mmstart.sh', shell=True)
os.remove(file_name)
subprocess.call('pip3 install -r requirements.txt', shell=True)
subprocess.call('bash mmdetection_install.sh', shell=True)