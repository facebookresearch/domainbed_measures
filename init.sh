#Copyright (c) Meta Platforms, Inc. and affiliates.
#All rights reserved.

#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.
echo "Creating virtual environment named dm"
python3 -m venv dm
source dm/bin/activate 

pip3 install torch torchvision torchaudio
pip install torch_scatter

pip install jsonlines
pip install matplotlib
pip install gdown
pip install pandas
pip install scipy==1.5.4
pip install scikit-learn==0.23.0
pip install wilds --upgrade
pip install submitit

# Dependencies for DIB
pip install skorch

echo "Adding DomainBed as a submodule at ${PWD}/DomainBed"
git submodule init
git submodule update
git submodule add git@github.com:facebookresearch/DomainBed.git