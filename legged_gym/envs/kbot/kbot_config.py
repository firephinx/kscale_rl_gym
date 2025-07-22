from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

import math

class KBotRoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.1] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'dof_left_hip_pitch_04' : math.radians(20.0),
           'dof_left_hip_roll_03': 0.,
           'dof_left_hip_yaw_03': 0.,
           'dof_left_knee_04': math.radians(50.0),
           'dof_left_ankle_02': math.radians(-30.0),   
           'dof_left_shoulder_pitch_03' : 0.,        
           'dof_left_shoulder_roll_03': math.radians(10.0),
           'dof_left_shoulder_yaw_02': 0.,
           'dof_left_elbow_02': math.radians(-90.0),
           'dof_left_wrist_00': 0.,
           'dof_right_hip_pitch_04' : math.radians(-20.0),
           'dof_right_hip_roll_03': 0.,
           'dof_right_hip_yaw_03': 0.,
           'dof_right_knee_04': math.radians(-50.0),
           'dof_right_ankle_02': math.radians(30.0),
           'dof_right_shoulder_pitch_03' : 0.,   
           'dof_right_shoulder_roll_03': math.radians(-10.0),
           'dof_right_shoulder_yaw_02': 0.,
           'dof_right_elbow_02': math.radians(90.0),
           'dof_right_wrist_00': 0.,
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 71
        num_privileged_obs = 74
        num_actions = 20


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_pitch': 100,
                     'hip_roll': 200,
                     'hip_yaw': 100,
                     'knee': 150,
                     'ankle': 40,
                     'shoulder_pitch': 100,
                     'shoulder_roll': 100,
                     'shoulder_yaw': 40,
                     'elbow': 40,
                     'wrist': 20,
                     }  # [N*m/rad]
        damping = {  'hip_pitch': 2,
                     'hip_roll': 26.387,
                     'hip_yaw': 3.419,
                     'knee': 8.654,
                     'ankle': 0.99,
                     'shoulder_pitch': 8.284,
                     'shoulder_roll': 8.257,
                     'shoulder_yaw': 0.945,
                     'elbow':1.266,
                     'wrist': 0.295
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kbot-headless/robot/robot.urdf'
        name = "kbot"
        foot_name = 'LEG_FOOT'
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 1.0
        only_positive_rewards = False

        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 10.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -20.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 1.0
            hip_pos = 0.0 #-1.0
            contact_no_vel = -2.0
            feet_swing_height = -10.0 #-0.2
            contact = 1.8

class KBotRoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'kbot'

  
