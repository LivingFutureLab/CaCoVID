
conda create -n cacovid_lmms python=3.10 -y
conda activate cacovid_lmms
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 #  -i https://pypi.tuna.tsinghua.edu.cn/simple # for H100
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 #  -i https://pypi.tuna.tsinghua.edu.cn/simple # for H20
pip install -r requirements.txt # -i https://pypi.tuna.tsinghua.edu.cn/simple