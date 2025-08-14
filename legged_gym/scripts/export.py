# legged_gym/scripts/export.py
#
# Export a trained rsl_rl policy as TorchScript and K-Infer binary.
# The exported model exposes the following signature:
#     init() -> carry
#     step(joint_angles, joint_velocities, command, carry) -> (actions, carry)
#
# Example usage: python legged_gym/scripts//export.py --task=kbot
# Or you can omit the checkpoint arg and it will use the latest checkpoint in the logs/kbot/ directory

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
import math

class TorchPolicyExporter(torch.nn.Module):
    """TorchScript-friendly stateless wrapper that works for FF, GRU, and LSTM nets."""

    def __init__(self, policy, normalizer=None):
        super().__init__()

        self.is_recurrent = policy.is_recurrent
        
        # Extract policy components
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy has neither actor nor student module.")

        # Set up RNN configuration
        if self.is_recurrent:
            self.rnn.cpu()
            self.num_layers = self.rnn.num_layers
            self.hidden_size = self.rnn.hidden_size
            rnn_name = type(self.rnn).__name__.lower()
            
            if "gru" in rnn_name:
                self.rnn_type = "gru"
                self.forward = self._forward_gru
            elif "lstm" in rnn_name:
                self.rnn_type = "lstm"
                self.forward = self._forward_lstm
                self.hidden_states = torch.zeros((2,self.num_layers,self.hidden_size))
                # self.hidden_states[0] = torch.tensor([2.3893e-02, -4.0182e-02, -1.6300e-01, -3.2848e-01, -8.9247e-02,
                #                             1.4373e-01, -8.0895e-02,  2.8819e-01, -6.0057e-02,  6.3056e-02,
                #                             -7.3454e-02, -2.3794e-01, -1.8860e-01, -5.7388e-03,  6.1775e-02,
                #                             2.9514e-02, -1.6111e-01,  1.5486e-02,  3.0160e-02, -3.6248e-02,
                #                             2.8146e-01, -2.4034e-01, -4.1731e-02, -2.3256e-04,  6.4057e-03,
                #                             -8.4956e-03,  2.0735e-01, -1.7650e-01, -1.3972e-01,  6.6886e-02,
                #                             -7.9494e-03, -3.4614e-01,  3.5153e-02,  4.1123e-02,  5.9506e-02,
                #                             1.0999e-01, -3.1818e-02, -1.0418e-01,  1.0993e-01, -3.5744e-01,
                #                             1.1790e-01,  1.5177e-01, -2.1752e-01, -7.5067e-01,  8.7959e-03,
                #                             -1.5438e-02, -2.4331e-02, -2.0172e-01, -1.1457e-02, -3.0508e-01,
                #                             2.0681e-03, -2.1112e-02,  2.6977e-01,  5.3435e-04,  7.5013e-03,
                #                             -8.7089e-02, -2.3484e-02, -1.2148e-01, -1.7802e-01, -6.0415e-02,
                #                             -1.0382e-01,  1.4228e-01,  2.5448e-03, -2.2938e-01])
                # self.hidden_states[1] = torch.tensor([ 0.0468, -0.2663, -0.2357, -0.7179, -0.0922,  0.5611, -0.4141,
                #                         2.2420, -0.6473,  0.1081, -1.7467, -0.3339, -0.7897, -0.1322,
                #                         0.4681,  0.0299, -0.5360,  0.1398,  1.1115, -0.9258,  0.5465,
                #                         -0.3131, -0.0929, -0.0044,  0.0344, -0.1050,  0.4760, -0.2022,
                #                         -0.3294,  0.9172, -0.1460, -0.6849,  0.0444,  0.0560,  0.6731,
                #                         0.1416, -0.2029, -0.6703,  0.7823, -0.6525,  0.2251,  0.1541,
                #                         -0.6548, -2.3410,  0.6449, -0.0830, -0.2681, -0.6403, -0.3193,
                #                         -0.6908,  0.0307, -0.0229,  0.2918,  0.5726,  0.7040, -0.0915,
                #                         -0.4894, -0.1332, -0.5981, -0.3617, -0.4222,  0.3750,  0.0777,
                #                         -0.2395])
            else:
                raise NotImplementedError(f"Unsupported RNN type: {rnn_name}")
                
            print(
                f"[DEBUG] RNN type: {self.rnn_type}, layers: {self.num_layers}, "
                f"hidden_size: {self.hidden_size}"
            )
        else:
            self.rnn_type = "ff"
            self.num_layers = 0
            self.hidden_size = 0
            self.forward = self._forward_ff

        # Set up normalizer
        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()

    def _forward_ff(self, obs: torch.Tensor, carry: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Feedforward forward pass (stateless for consistency)."""
        obs = self.normalizer(obs)
        actions = self.actor(obs)
        return actions, actions

    def _forward_gru(self, obs: torch.Tensor, carry: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """GRU forward pass with stateless carry management."""
        obs = self.normalizer(obs)
        
        # carry shape: (1, num_layers, hidden_size)
        if carry.dim() != 3 or carry.size(0) != 1:
            raise RuntimeError(f"Expected GRU carry shape (1, {self.num_layers}, {self.hidden_size}), got {carry.shape}")
        
        # Extract hidden state: (num_layers, 1, hidden_size)
        hidden = carry[0].unsqueeze(1)
        # Input needs to be 3D for RNN: (seq_len=1, batch_size=1, input_size)
        x = obs.unsqueeze(0).unsqueeze(0)
        out, new_hidden = self.rnn(x, hidden)
        actions = self.actor(out.squeeze(0).squeeze(0))
        # Reshape hidden back to carry format: (1, num_layers, hidden_size)
        new_carry = new_hidden.squeeze(1).unsqueeze(0)
        return actions, new_carry

    def _forward_lstm(self, obs: torch.Tensor, carry: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """LSTM forward pass with stateless carry management."""
        obs = self.normalizer(obs)
        
        # Extract hidden and cell states: (num_layers, 1, hidden_size)
        h = self.hidden_states[0].unsqueeze(1)
        c = self.hidden_states[1].unsqueeze(1)
        # Input needs to be 3D for RNN: (seq_len=1, batch_size=1, input_size)
        x = obs.unsqueeze(0).unsqueeze(0)
        out, (new_h, new_c) = self.rnn(x, (h, c))
        actions = self.actor(out.squeeze(0).squeeze(0))
        # Reshape states back to carry format: (2, num_layers, hidden_size)
        self.hidden_states = torch.stack(
            (new_h.squeeze(1), new_c.squeeze(1)), dim=0
        )
        return actions, actions

    def get_carry_shape(self, num_joints: int) -> Tuple[int, ...]:
        """Get the shape of the carry tensor for this policy."""
        if self.rnn_type == "ff":
            return (num_joints,)
        elif self.rnn_type == "gru":
            return (1, self.num_layers, self.hidden_size)
        elif self.rnn_type == "lstm":
            return (num_joints,)
            #return (2, self.num_layers, self.hidden_size)
        else:
            raise RuntimeError(f"Unknown RNN type: {self.rnn_type}")

    def get_initial_carry(self, num_joints: int) -> torch.Tensor:
        """Get the initial (zero) carry tensor for this policy."""
        return torch.zeros(self.get_carry_shape(num_joints))

    def get_traced_module(self):
        """Return a scripted (CPU, eval) version of this module."""
        self.to("cpu").eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return torch.jit.script(self)


args = get_args()
env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

env_cfg.env.num_envs = 1
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

#olicy_nn = ppo_runner.alg.actor_critic

try:
    policy_nn = ppo_runner.alg.policy
except AttributeError:
    policy_nn = ppo_runner.alg.actor_critic

# Optionally include normalizer if present
normalizer = getattr(ppo_runner, "obs_normalizer", None)
if normalizer is None:
    normalizer = torch.nn.Identity()
else:
    normalizer = copy.deepcopy(normalizer).cpu().eval()

exporter = TorchPolicyExporter(policy_nn, normalizer)

exporter.to("cpu").eval()
for p in exporter.parameters():
    p.requires_grad_(False)

ts_policy = exporter

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

_INIT_JOINT_POS = torch.tensor(
        [
            math.radians(10.0),  # dof_left_hip_pitch_04
            0.0,  # dof_left_hip_roll_03
            0.0,  # dof_left_hip_yaw_03
            math.radians(30.0),  # dof_left_knee_04
            math.radians(-20.0),  # dof_left_ankle_02
            0.0,  # dof_left_shoulder_pitch_03
            math.radians(10.0),  # dof_left_shoulder_roll_03
            0.0,  # dof_left_shoulder_yaw_02
            math.radians(-20.0),  # dof_left_elbow_02
            0.0,  # dof_left_wrist_00
            math.radians(-10.0),  # dof_right_hip_pitch_04
            0.0,  # dof_right_hip_roll_03
            0.0,  # dof_right_hip_yaw_03
            math.radians(-30.0),  # dof_right_knee_04
            math.radians(20.0),  # dof_right_ankle_02
            0.0,  # dof_right_shoulder_pitch_03
            math.radians(-10.0),  # dof_right_shoulder_roll_03
            0.0,  # dof_right_shoulder_yaw_02
            math.radians(20.0),  # dof_right_elbow_02
            0.0,  # dof_right_wrist_00
            # 0.0,  # dof_left_hip_pitch_04
            # 0.0,  # dof_left_hip_roll_03
            # 0.0,  # dof_left_hip_yaw_03
            # 0.0,  # dof_left_knee_04
            # math.radians(-5.0),  # dof_left_ankle_02
            # 0.0,  # dof_left_shoulder_pitch_03
            # math.radians(10.0),  # dof_left_shoulder_roll_03
            # 0.0,  # dof_left_shoulder_yaw_02
            # math.radians(-90.0),  # dof_left_elbow_02
            # 0.0,  # dof_left_wrist_00
            # 0.0,  # dof_right_hip_pitch_04
            # 0.0,  # dof_right_hip_roll_03
            # 0.0,  # dof_right_hip_yaw_03
            # 0.0,  # dof_right_knee_04
            # math.radians(5.0),  # dof_right_ankle_02
            # 0.0,  # dof_right_shoulder_pitch_03
            # math.radians(-10.0),  # dof_right_shoulder_roll_03
            # 0.0,  # dof_right_shoulder_yaw_02
            # math.radians(90.0),  # dof_right_elbow_02
            # 0.0,  # dof_right_wrist_00
        ]
    )

_JOINT_LIMITS = torch.tensor(
        [[-1.0,  2.2],
        [-0.2,  2.25],
        [-1.55,  1.55],
        [ 0.01,  2.7],
        [-1.25,  0.22],
        [-1.39,  3.14],
        [-0.34,  1.65],
        [-1.65,  1.65],
        [-2.47,  -0.01],
        [-1.74,  1.74],
        [-2.21,  1.04],
        [-2.26,  0.2],
        [-1.57,  1.57],
        [-2.7,  -0.01],
        [-0.22,  1.25],
        [-3.14,  1.39],
        [-1.65,  0.34],
        [-1.65,  1.65],
        [ 0.01,  2.47],
        [-1.74,  1.74]]
    )

NUM_JOINTS = len(joint_names)
NUM_COMMANDS = 3
NUM_ACTIONS = 14

# Get carry shape from the exporter
CARRY_SHAPE = exporter.get_carry_shape(NUM_ACTIONS)

ACTION_SCALE = 0.25

cmd_scale = torch.tensor([2.0, 2.0, 0.25])
dof_pos_scale = 1.0
dof_vel_scale = 0.05
ang_vel_scale = 0.25

def construct_obs_rnn(
    projected_gravity: torch.Tensor,
    joint_angles: torch.Tensor,
    joint_angular_velocities: torch.Tensor,
    command: torch.Tensor,
    gyroscope: torch.Tensor,
    carry: torch.Tensor,
) -> torch.Tensor:
    scaled_projected_gravity = projected_gravity / 9.81
    scaled_gyro = gyroscope * ang_vel_scale
    scaled_command = command * cmd_scale
    offset_joint_angles = (joint_angles - _INIT_JOINT_POS) * dof_pos_scale
    scaled_joint_angular_velocities = joint_angular_velocities * dof_vel_scale
    obs = torch.cat(
        (
            scaled_projected_gravity,
            scaled_gyro,
            scaled_command,
            offset_joint_angles,
            scaled_joint_angular_velocities,
            carry
        ),
        dim=-1,
    )
    return obs

def construct_obs_ff(
    projected_gravity: torch.Tensor,
    joint_angles: torch.Tensor,
    joint_angular_velocities: torch.Tensor,
    command: torch.Tensor,
    gyroscope: torch.Tensor,
    carry: torch.Tensor,
) -> torch.Tensor:
    obs = construct_obs_rnn(projected_gravity, joint_angles, joint_angular_velocities, command, gyroscope, carry)
    #obs = torch.cat((obs, carry), dim=-1)
    return obs

# Recurrent or feedforward logic split out here so it doesn't get traced and err out on carry and matmul shape mismatches
# Difference is that obs_rnn does not add the carry to the obs (holds previous action for purely feedforward policies)
if exporter.is_recurrent:
    construct_obs = construct_obs_rnn
else:
    construct_obs = construct_obs_ff

def _step_fn(
    projected_gravity: torch.Tensor,
    joint_angles: torch.Tensor,
    joint_angular_velocities: torch.Tensor,
    command: torch.Tensor,
    gyroscope: torch.Tensor,
    carry: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Policy step."""
    obs = construct_obs(projected_gravity, joint_angles, joint_angular_velocities, command, gyroscope, carry)

    actions, new_carry = ts_policy(obs, carry)
    
    clamped_actions = torch.cat(
        (
            (actions[:6] * ACTION_SCALE) + _INIT_JOINT_POS[:6],
            _INIT_JOINT_POS[6:8],
            (actions[6] * ACTION_SCALE) + _INIT_JOINT_POS[8],
            _INIT_JOINT_POS[9],
            (actions[7:13] * ACTION_SCALE) + _INIT_JOINT_POS[10:16],
            _INIT_JOINT_POS[16:18],
            (actions[13] * ACTION_SCALE) + _INIT_JOINT_POS[18],
            _INIT_JOINT_POS[19],
        ),
        dim=-1,
    )
    for i in range(NUM_JOINTS):
        clamped_actions[i] = torch.clamp(clamped_actions[i], _JOINT_LIMITS[i, 0], _JOINT_LIMITS[i, 1])

    return clamped_actions, new_carry
  
def _init_fn() -> torch.Tensor:
    return exporter.get_initial_carry(*CARRY_SHAPE)

step_args = (
    torch.zeros(3), # projected_gravity
    torch.zeros(NUM_JOINTS),
    torch.zeros(NUM_JOINTS),
    torch.zeros(NUM_COMMANDS), 
    torch.zeros(3), # gyroscope
    torch.zeros(*CARRY_SHAPE),
)

step_fn = torch.jit.trace(_step_fn, step_args)#, check_trace=False)
init_fn = torch.jit.trace(_init_fn, ())

# ----------------------------------------------------------------------------
# Export to ONNX (via kinfer) and package
# ----------------------------------------------------------------------------

#joint_names = list(env.unwrapped.scene["robot"].data.joint_names)
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
