# Environment setup
conda create --name hand2gripper python=3.10
conda activate hand2gripper
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Submodules
cd hand2gripper/submodules/
# Hand2Gripper WiLoR
git clone https://github.com/yutian929/Hand2Gripper_WiLoR.git
cd Hand2Gripper_WiLoR
pip install -r requirements.txt
pip install "numpy<2"
pip install -e .
cd hand2gripper_wilor/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
echo -e "\033[41mIt is also required to download MANO model from MANO website (https://mano.is.tue.mpg.de/) . Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and place the right hand model MANO_RIGHT.pkl under the mano_data/ folder. Note that MANO model falls under the MANO license.\033[0m"
read -p -e "\033[41mPress Enter to continue\033[0m"
cd ../..
# Hand2Gripper HACO
git clone https://github.com/yutian929/Hand2Gripper_HACO.git
cd Hand2Gripper_HACO
pip install -r requirements.txt
pip install "numpy<2"
pip install -e .
bash scripts/download_base_data.sh
# echo -e "\033[41mWhile using HACO, please set the environment variable HACO_BASE_DATA_PATH to the base data path\033[0m"
# read -p -e "\033[41mPress Enter to continue\033[0m"
cd ..
# Hand2Gripper XXX