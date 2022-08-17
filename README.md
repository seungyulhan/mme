# Max-Min Entropy (MME) Framework

This repository is an implementation of A Max-Min Entropy Framework for Reinforcement Learning (NeurIPS 2021)
```
@article{han2021max,
  title={A Max-Min Entropy Framework for Reinforcement Learning},
  author={Han, Seungyul and Sung, Youngchul},
  journal={arXiv preprint arXiv:2106.10517},
  year={2021}
}
```

## Dependencies

The implementation is based on [the source code](https://github.com/rail-berkeley/softlearning) of soft actor-critic [SAC](https://github.com/haarnoja/sac)

This implementation requires Anaconda / rllab / Mujoco / Tensorflow.

## Getting Started

1. Clone rllab [rllab](https://github.com/rll/rllab):

```
cd <install_path>
git clone https://github.com/rll/rllab.git
cd rllab
git checkout b3a28992eca103cab3cb58363dd7a4bb07f250a0
export PYTHONPATH=$<install_path>:${PYTHONPATH}
```

2. Create conda environment and add path:
```
conda create -n mme python=3.6
export PATH="/home/<user_name>/anaconda3/envs/dac/bin:$PATH"
```

3. Install libraries and packages:
```
sudo apt-get install python3-pip mpich libopenmpi-dev libgl-dev libglu-dev libxrandr-dev libxinerama-dev libxi-dev libxcursor-dev
conda activate mme

pip install numpy scipy path.py python-dateutil joblib==0.10.3 mako ipywidgets numba flask pygame h5py matplotlib mpi4py torchvision==0.1.6 pandas Pillow atari-py ipdb boto3 PyOpenGL nose2 pyzmq tqdm msgpack-python mujoco_py==0.5.7 cached_property line_profiler cloudpickle Cython redis git+https://github.com/Theano/Theano.git@adfe319ce6b781083d8dc3200fb4481b00853791#egg=Theano git+https://github.com/neocxi/Lasagne.git@484866cf8b38d878e92d521be445968531646bb8#egg=Lasagne plotly git+https://github.com/rll/rllab.git@b3a28992eca103cab3cb58363dd7a4bb07f250a0#egg=rllab git+https://github.com/openai/gym.git@v0.7.4#egg=gym awscli pyglet jupyter progressbar2 tensorflow==1.4 numpy-stl==2.2.0 nibabel==2.1.0 pylru==1.0.9 hyperopt polling gtimer git+https://github.com/neocxi/prettytensor.git pyprind scikit-learn==0.20.0
```

## Examples for Training Agent
1. MME in pure exploration

```
python -m examples.run_mme --env=2Dmaze-cont --task pure --alpha_pi 1 --alpha_Q 0.5 --gamma 0.999
```

2. MME in SparseMujoco tasks

```
python -m examples.run_mme --env=half-cheetah --task sparse --alpha_pi 0.02 --alpha_Q 2.0 --gamma 0.99
```

3. DE-MME in DelayedMujoco tasks

```
python -m examples.run_demme --env=half-cheetah --task delayed --alpha_pi 0.2 --alpha_Q 2.0 --gamma 0.99
```
