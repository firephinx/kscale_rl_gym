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
        num_observations = 66
        num_privileged_obs = 74
        num_actions = 20

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        #num_curriculum_levels = 10
        max_curriculum = 1.
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]


    class domain_rand(LeggedRobotCfg.domain_rand):
        curriculum = False
        #num_curriculum_levels = 10
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_link_masses = True
        added_mass_range = [-0.1, 0.1]
        randomize_gains = True
        randomize_gains_fraction = 0.05
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
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kbot-headless-full-collisions/robot.urdf'
        name = "kbot"
        foot_name = 'LEG_FOOT'
        knee_name = 'Femur'
        imu_name = "imu"
        arm_names = ["shoulder", "elbow", "wrist"]
        hip_names = ['hip_roll', 'hip_yaw']
        ankle_names = ['ankle']
        knee_names = ['knee']
        #penalize_contacts_on = ["knee", "hip"]
        #terminate_after_contacts_on = ["knee", "hip", "base", "shoulder", "wrist", "Bayonet"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 1.0
        only_positive_rewards = True
        min_dist = 0.2
        max_dist = 0.5

        max_contact_force = 400  # forces above this value are penalized
        feet_swing_height = 0.1

        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 10.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -20.0
            dof_acc = -1e-7
            dof_vel = -5e-4
            torques = -0.00001
            action_rate = -0.005
            dof_pos_limits = -10.0
            alive = 7.0
            feet_swing_height = -20.0 #-0.2
            contact = 0.18
            contact_no_vel = -0.2

            arm_deviation = -1.0 # -0.1
            hip_deviation = -1.0
            ankle_deviation = -0.1
            ankle_pos_limits = -10.0

            feet_contact_forces = -0.01
            flat_feet = 0.0 #-0.1 # -2.0
            feet_air_time = 0.0 #0.25
            collision = 0.0
            foot_slip = 0.0 #-0.1
            action_smoothness = 0.0
            knee_distance = 0.0

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
        log_dir = 'logs/kbot/'

  
