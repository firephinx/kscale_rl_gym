
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
import torch

class KBot(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        #noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[:3] = noise_scales.gravity * noise_level
        noise_vec[3:6] = 0. # commands
        noise_vec[6:6+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[6+self.num_actions:6+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[6+2*self.num_actions:6+3*self.num_actions] = 0. # previous actions
        #noise_vec[9+3*self.num_actions:6+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
 
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_quat = self.feet_state[:, :, 3:7]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_quat = self.feet_state[:, :, 3:7]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.projected_gravity,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)

        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        for i in range(self.feet_num):
            is_stance = (self.leg_phase[:, i] < 0.55) 
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
        return res

    def _reward_contact_stand_still(self):
        left_contact = self.contact_forces[:, self.feet_indices[0], 2] > 1
        right_contact = self.contact_forces[:, self.feet_indices[1], 2] > 1
        
        return left_contact * right_contact * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2)) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
    
    def _reward_feet_contact_forces(self):
        """Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force
            ).clip(0, 400),
            dim=1,
        )

    def _reward_feet_height(self):
        # Penalize base height away from target
        feet_height = self.feet_pos[:, :, 2]
        dif = torch.abs(feet_height - self.cfg.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target 
        dif *= torch.norm(self.commands[:, :2], dim=1) > 0.1  #no reward for zero command
        return torch.clip(dif - 0.02, min=0.) # target - 0.02 ~ target + 0.02 is acceptable 

    def _reward_feet_ori(self):
        left_quat = self.feet_quat[:, 0, :]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.feet_quat[:, 1, :]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

    def _reward_slippage(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            res += torch.norm(self.feet_vel[:,i,:], dim=-1) * (torch.norm(self.contact_forces[:, self.feet_indices[i], :], dim=-1) > 1.) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
        return res


############################# DEPRECATED REWARD FUNCTIONS ##########################################

    # def _reward_feet_swing_height(self):
    #     contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
    #     pos_error = torch.square(self.feet_pos[:, :, 2] - self.cfg.rewards.feet_swing_height) * ~contact
    #     return torch.sum(pos_error, dim=(1))
    
    # def _reward_flat_feet(self):
    #     # 1) quats → euler, keep roll & pitch only
    #     right_foot_rp = get_euler_xyz_in_tensor(self.feet_quat[:,0,:])[:,:2]
    #     left_foot_rp = get_euler_xyz_in_tensor(self.feet_quat[:,1,:])[:,:2]

    #     tgt = torch.tensor([0.0,0.0]).to(device=self.device)
    #     rp_err = torch.abs(left_foot_rp - tgt).sum(axis=-1) + torch.abs(right_foot_rp - tgt).sum(axis=-1)
    #     return rp_err

    

    # def _reward_action_smoothness(self):
    #     """Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
    #     This is important for achieving fluid motion and reducing mechanical stress.
    #     """
    #     term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    #     term_2 = torch.sum(
    #         torch.square(self.actions + self.last_last_actions - 2 * self.last_actions),
    #         dim=1,
    #     )
    #     term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
    #     return term_1 + term_2 + term_3