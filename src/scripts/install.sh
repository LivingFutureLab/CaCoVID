
conda create -n cacovid python=3.10 -y
conda activate cacovid
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 #  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt # -i https://pypi.tuna.tsinghua.edu.cn/simple