pip install -U cython

pip install -U ninja sparsehash h5py pyyaml tensorboard tensorboardx wandb yapf addict \
  einops scipy plyfile termcolor timm ftfy regex tqdm matplotlib black \
  open3d

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install torch-cluster torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

pip install -U torch-geometric

pip install -U spconv-cu124

pip install -U git+https://github.com/octree-nn/ocnn-pytorch.git
pip install -U git+https://github.com/openai/CLIP.git
#pip install -U git+https://github.com/Dao-AILab/flash-attention.git
pip install flash-attn --no-build-isolation

cd libs/pointops
pip install .

cd ../pointgroup_ops
pip install .
