# Point Cept

## Environment

### CUDA

<https://developer.nvidia.com/cuda-toolkit-archive>

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sudo sh cuda_12.4.1_550.54.15_linux.run
```

### CUDNN

<https://developer.nvidia.com/cudnn-downloads>

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.10.2/local_installers/cudnn-local-repo-ubuntu2004-9.10.2_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2004-9.10.2_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2004-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
```

### NCCL

<https://developer.nvidia.com/nccl/nccl-download>

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt install libnccl2=2.27.5-1+cuda12.4 libnccl-dev=2.27.5-1+cuda12.4
```

## Setup

```bash
conda create -n pc python=3.10
conda activate pc
./setup.sh
```

## Run

```bash
python demo.py
```

## Enjoy it~
