pip install -U ninja

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -U h5py pyyaml sharedarray tensorboard \
  tensorboardx yapf addict einops scipy plyfile termcolor timm

pip install torch-cluster torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

pip install -U torch-geometric

pip install -U spconv-cu118

pip install -U ftfy regex tqdm
pip install -U git+https://github.com/openai/CLIP.git

cd libs/pointops
TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9" python setup.py install

pip install -U open3d

pip install -U flash-attn --no-build-isolation
