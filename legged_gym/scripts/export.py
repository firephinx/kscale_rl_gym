# scripts/reinforcement_learning/rsl_rl/export.py
#
# Export a trained rsl_rl policy as TorchScript and K-Infer binary.
# The exported model exposes the following signature:
#     init() -> carry
#     step(joint_angles, joint_velocities, command, carry) -> (actions, carry)
#
# Example usage: python scripts/reinforcement_learning/rsl_rl/export.py --task=Isaac-Velocity-Rough-Kbot-v0 --checkpoint ~/Github/IsaacLab/logs/rsl_rl/kbot_rough/[path_to_checkpoint].pt

import argparse
import copy
from pathlib import Path

import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import torch
import numpy as np
from kinfer.export.pytorch import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata
from typing import Tuple

args = get_args()
env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

env_cfg.env.num_envs = 1
env_cfg.terrain.num_rows = 5
env_cfg.terrain.num_cols = 5
env_cfg.terrain.curriculum = False
env_cfg.noise.add_noise = False
env_cfg.domain_rand.randomize_friction = False
env_cfg.domain_rand.push_robots = False

env_cfg.env.test = True

# prepare environment
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
obs = env.get_observations()
# load policy
train_cfg.runner.resume = True
ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
#policy = ppo_runner.get_inference_policy(device=env.device)
#policy_nn = ppo_runner.alg.actor_critic
#actor_net = copy.deepcopy(policy_nn.actor).cpu().eval()

try:
    policy_nn = ppo_runner.alg.policy
except AttributeError:
    policy_nn = ppo_runner.alg.actor_critic

#actor_net = copy.deepcopy(policy_nn).cpu().eval()
if hasattr(policy_nn, "actor"):
    actor_net = copy.deepcopy(policy_nn.actor).cpu().eval()
    memory_net = copy.deepcopy(policy_nn.memory_a).cpu().eval()
elif hasattr(policy_nn, "student"):
    actor_net = copy.deepcopy(policy_nn.student).cpu().eval()
    memory_net = copy.deepcopy(policy_nn.memory_a).cpu().eval()
else:
    raise RuntimeError("Unsupported policy object – cannot locate actor network")

normalizer = getattr(ppo_runner, "obs_normalizer", None)
if normalizer is None:
    normalizer = torch.nn.Identity().cpu().eval()
else:
    normalizer = copy.deepcopy(normalizer).cpu().eval()

class ActorWrapper(torch.nn.Module):
    """Wraps (normalizer → actor) into a single forward pass."""

    def __init__(self, actor, memory, norm):
        super().__init__()
        self.actor = actor
        self.memory = memory
        self.norm = norm
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # noqa: D401 – simple wrapper
        return self.actor(self.memory(self.norm(obs))).squeeze()
        

wrapper = ActorWrapper(actor_net, memory_net, normalizer).eval()
for p in wrapper.parameters():
    p.requires_grad = False

num_joints = 20
CARRY_SHAPE: Tuple[int, ...] = (num_joints,)
NUM_COMMANDS = 3

def _init_fn() -> torch.Tensor:  # noqa: D401 – concise docstring
    """Returns the initial carry tensor (all zeros)."""
    return torch.zeros(CARRY_SHAPE)

cmd_scale = torch.tensor([1.0, 1.0, 0.25])
action_scale = torch.tensor(0.25)
dof_pos_scale = torch.tensor(1.0)
dof_vel_scale = torch.tensor(0.05)
default_angles = torch.tensor([0.3491, 0.0, 0.0, 0.8727, -0.5236, 0.0, 0.1745, 0.0, -1.5708, 0.0, -0.3491, 0.0, 0.0, -0.8727, 0.5236, 0.0, -0.1745, 0.0, 1.5708, 0.0])
num_actions = torch.tensor(20)

def _step_fn(
    projected_gravity: torch.Tensor,
    joint_angles: torch.Tensor,
    joint_angular_velocities: torch.Tensor,
    command: torch.Tensor,
    carry: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Policy step."""
    
    cmd = command * cmd_scale
    joint_angles = (joint_angles - default_angles) * dof_pos_scale
    joint_angular_velocities = joint_angular_velocities * dof_vel_scale

    obs = torch.cat((projected_gravity, cmd, joint_angles, joint_angular_velocities, carry), dim=-1)
    actions = wrapper(obs)

    return actions * action_scale + default_angles, actions

step_fn = torch.jit.trace(
    _step_fn,
    (
        torch.zeros(3),
        torch.zeros(num_joints),
        torch.zeros(num_joints),
        torch.zeros(NUM_COMMANDS),
        torch.zeros(*CARRY_SHAPE),
    ),
    check_trace=False
)

init_fn = torch.jit.trace(_init_fn, ())

# ----------------------------------------------------------------------------
# Export to ONNX (via kinfer) and package
# ----------------------------------------------------------------------------

joint_names = ['dof_left_hip_pitch_04',
                'dof_left_hip_roll_03',
                'dof_left_hip_yaw_03',
                'dof_left_knee_04',
                'dof_left_ankle_02',
                'dof_left_shoulder_pitch_03',
                'dof_left_shoulder_roll_03',
                'dof_left_shoulder_yaw_02',
                'dof_left_elbow_02',
                'dof_left_wrist_00',
                'dof_right_hip_pitch_04',
                'dof_right_hip_roll_03',
                'dof_right_hip_yaw_03',
                'dof_right_knee_04',
                'dof_right_ankle_02',
                'dof_right_shoulder_pitch_03',
                'dof_right_shoulder_roll_03',
                'dof_right_shoulder_yaw_02',
                'dof_right_elbow_02',
                'dof_right_wrist_00']
metadata = PyModelMetadata(
    joint_names=joint_names,
    num_commands=NUM_COMMANDS,
    carry_size=list(CARRY_SHAPE),
)

init_onnx = export_fn(init_fn, metadata)
step_onnx = export_fn(step_fn, metadata)

kinfer_blob = pack(init_fn=init_onnx, step_fn=step_onnx, metadata=metadata)

output_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies', 'model.kinfer')
with open(output_path, "wb") as f:
    f.write(kinfer_blob)

print(f"[OK] Export completed → {output_path}")

EXPORT_POLICY = True

if EXPORT_POLICY:
    # Optionally also save the JIT + ONNX versions (same directory /exported)
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    print('Exported policy as jit script to: ', path)
