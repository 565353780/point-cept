cd ..
git clone --depth 1 --recursive -b v2.8.3 https://github.com/Dao-AILab/flash-attention.git

conda install -c conda-forge sparsehash -y

pip install -U cython

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -U packaging ninja sparsehash h5py pyyaml sharedarray tensorboard \
  tensorboardx wandb yapf addict einops scipy plyfile termcolor timm ftfy regex \
  tqdm matplotlib black open3d

pip install torch-cluster torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

pip install -U torch-geometric

pip install -U spconv-cu124

#pip install -U git+https://github.com/octree-nn/ocnn-pytorch.git
#pip install -U git+https://github.com/openai/CLIP.git
cd flash-attention
python setup.py install

cd ../point-cept/libs/pointops
pip install .

cd ../pointgroup_ops
pip install .
