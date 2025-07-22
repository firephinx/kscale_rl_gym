from setuptools import find_packages
from distutils.core import setup

setup(name='kscale_rl_gym',
      version='1.0.0',
      author='K Scale Labs',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='inquire@kscale.dev',
      description='Template RL environments for KScale Robots',
      install_requires=['isaacgym', 'rsl-rl', 'matplotlib', 'numpy==1.20', 'tensorboard', 'mujoco==3.2.3', 'pyyaml'])
