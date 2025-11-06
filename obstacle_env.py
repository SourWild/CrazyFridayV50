# obstacle_env.py

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# =======================
# Env / 环境
# =======================
class ObstacleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # =======================
    # Initialization / 初始化
    # =======================
    def __init__(self, xml_path, obstacle_mode="dynamic", render_mode="human"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.time = 0.0
        self.obstacle_mode = obstacle_mode
        self.render_mode = render_mode
        self.viewer = None  # passive viewer (GUI)
        self.cam = None     # for rgb_array mode
        self.max_episode_steps =100_000  # enforce 10k-step horizon
        self.episode_step = 0
       
        # =======================
        # Action space (8dim): 
        # 动作空间(8dim): agent 的 8 个 qpos
        # =======================
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        
        # =======================
        # Observation space (24dim): Agent robotic arm with 8 joints qpos, 8 joints qvel, observer with 8 joints qpos
        # 观测空间（24dim）： agent机械臂8个关节qpos，8个关节qvel，obstacle 8个关节 qpos
        # =======================
        obs_dim = 8*2 + 8
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # =======================
        # PD control parameters
        # PD 控制参数
        # ======================= 
        self.Kp = 30.0
        self.Kd = 10.0
        self.dt = 0.002 
        
        # =======================
        # Parameters under different movement patterns of obstacles
        # 障碍物不同运动 pattern 下的参数
        # =======================
        
        # Parameters under dynamic movement pattern of obstacles
        # 障碍物 dynamic pattern 下的参数
        self.A_list = [0, 0, 0, 0, 0, 0, 0, 0.1] # amp / 振幅
        self.f_list = [0, 0, 0, 0, 0, 0, 0, 0.5] # freq / 频率        
  
    
        
    # =======================
    #  / 归一化映射
    # =======================
    def action_to_qpos(self, action):
        joint_range = self.model.jnt_range[:8]
        q_min = joint_range[:, 0]
        q_max = joint_range[:, 1]
        q_target = q_min + (action + 1) * 0.5 * (q_max - q_min)
        return q_target
    
    
    # =======================
    # Check collision for points / 得分检测
    # =======================
    
    def _check_collision_point(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get geom ID
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            # Get contype
            contype1 = self.model.geom_contype[geom1_id]
            contype2 = self.model.geom_contype[geom2_id]

            # Get body name
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            # name1 = self.model.body_id2name(body1_id)
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            # name2 = self.model.body_id2name(body2_id)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            # Agent ee ↔ Obstacle torso
            cond1 = (contype1 == 4 and "agent" in name1 and contype2 == 17 and "obstacle" in name2)
            cond2 = (contype2 == 4 and "agent" in name2 and contype1 == 17 and "obstacle" in name1)
            # Obstacle ee ↔ Agent torso
            cond3 = (contype1 == 8 and "obstacle" in name1 and contype2 == 17 and "agent" in name2)
            cond4 = (contype2 == 8 and "obstacle" in name2 and contype1 == 17 and "agent" in name1)
            

            if ( cond1 or cond2 ) and ( cond3 or cond4 ): # both bull point / 双方同时得分
                return 3
            elif  cond1 or cond2 : # agent  bull point / agent 得分
                return 2
            elif  cond3 or cond4 : # obstacle  bull point / obstacle 得分
                return 1
            else :
                pass
        return 0
    
    # =======================
    # Check collision for no points /不得分情况的碰撞检测
    # =======================
    
    def _check_collision_no_point(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get geom ID
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            # Get contype
            contype1 = self.model.geom_contype[geom1_id]
            contype2 = self.model.geom_contype[geom2_id]

            # Get body name
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            # name1 = self.model.body_id2name(body1_id)
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            # name2 = self.model.body_id2name(body2_id)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            # Agent ee ↔ Obstacle arm
            cond1 = (contype1 == 4 and "agent" in name1 and contype2 == 18 and "obstacle" in name2)
            cond2 = (contype2 == 4 and "agent" in name2 and contype1 == 18 and "obstacle" in name1)
            # Obstacle ee ↔ Agent arm
            cond3 = (contype1 == 8 and "obstacle" in name1 and contype2 == 18 and "agent" in name2)
            cond4 = (contype2 == 8 and "obstacle" in name2 and contype1 == 18 and "agent" in name1)
            # Agent arm ↔ Obstacle arm
            cond5 = (contype1 == 18 and "agent" in name1 and contype2 == 18 and "obstacle" in name2)
            cond6 = (contype2 == 18 and "agent" in name2 and contype1 == 18 and "obstacle" in name1)

            if ( cond1 or cond2 ) or ( cond3 or cond4 ) or ( cond5 or cond6 ): # collision for no points / 不得分的碰撞
                return True
            else :
                pass
            
        return False
    
    # =======================
    # Switching obstacle movement patterns / 切换障碍物运动 pattern
    # =======================     
    def set_obstacle_mode(self, mode: str):
        # 'dynamic' or 'static' or 'periodic' or 'reactive'
        assert mode in ['static','none', 'periodic','reactive']
        # static：Control the end  effector at the initial position / 控制末端在初始位置
        # dynamic：Periodic motion based on joint qpos / 基于关节 qpos 的周期运动
        # periodic：Periodic motion of end effector / 末端的周期运动
        # reactive：Avoidance / 躲避
        self.obstacle_mode = mode
    
    # =======================
    # Agent and obstacle control / agent 和 bstacle 控制
    # ======================= 
    def step(self, action):
        
        # =======================
        # Agent control / agent 控制
        # =======================
        # 关节索引范围（8 个）
        joint_ids = np.arange(0, 8)
        actutator_ids = np.arange(0, 8)

        # 当前状态
        qpos_now = self.data.qpos[joint_ids]
        qvel_now = self.data.qvel[joint_ids]
        
        action = np.clip(action, -1, 1)  # 限制到[-1,1]
        qpos_target = self.action_to_qpos(action)
        qvel_target = 0.0

        
        # PD 控制扭矩
        torque = self.Kp * (qpos_target - qpos_now) + self.Kd * (qvel_target - qvel_now)

        # 限幅
        ctrl_min = self.model.actuator_ctrlrange[actutator_ids, 0]
        ctrl_max = self.model.actuator_ctrlrange[actutator_ids, 1]
        self.data.ctrl[actutator_ids] = np.clip(torque, ctrl_min, ctrl_max)

        # =======================
        # Obstacle control of different movement patterns / 不同运动模式下的 obstacle 控制
        # =======================
        # 关节索引范围（8 个）
        joint_ids = np.arange(12, 20)
        actutator_ids = np.arange(8, 16)

        if self.obstacle_mode == 'periodic':

            # 当前状态
            qpos_now = self.data.qpos[joint_ids]
            qvel_now = self.data.qvel[joint_ids]

            # 振幅与频率
            A = np.array(self.A_list)
            f = np.array(self.f_list)

            # 相位偏移（i * π/4）
            phase = np.arange(8) * np.pi / 4

            # 目标位置与速度
            qpos_target = A * np.sin(2 * np.pi * f * self.time + phase)
            qvel_target = 2 * np.pi * f * A * np.cos(2 * np.pi * f * self.time + phase)

            # PD 控制扭矩
            torque = self.Kp * (qpos_target - qpos_now) + self.Kd * (qvel_target - qvel_now)

            # 限幅
            ctrl_min = self.model.actuator_ctrlrange[actutator_ids, 0]
            ctrl_max = self.model.actuator_ctrlrange[actutator_ids, 1]
            self.data.ctrl[actutator_ids] = np.clip(torque, ctrl_min, ctrl_max)
            
        elif self.obstacle_mode == 'none':
            # Only gravity acts, and the output torque is zero / 只有重力作用，输出扭矩皆为 0
            self.data.ctrl[actutator_ids] = 0 
             
        
        elif self.obstacle_mode == 'reactive':
            # Avoidance pattern
            # 躲避模式
            
            # Control a sliding joint
            # 控制一个滑动关节
            # Current sliding joint position and velocity
            # 当前滑动关节位置和速度
            q_slide = self.data.qpos[15] 
            qd_slide = self.data.qvel[15]
            # If the agent is within 3 meters, trigger avoidance
            # 如果agent距离3米内，触发躲避            
            pos7 = self.data.xpos[7]    # shape (3,), [x, y, z]
            pos15 = self.data.xpos[15]  # shape (3,), [x, y, z]           
            if abs(pos7[1] - pos15[1]) < 2:                
                delta_pos_y = pos7[1] - pos15[1] - 3                                
                # target position
                # 目标位置    
                q_slide = np.array(q_slide)
                q_target = q_slide - delta_pos_y

                # Expected velocity is 0
                # 期望速度为 0
                qd_target = 0.0  
                # PD gain
                # PD 增益
                Kp = 100.0
                Kd = 10.0
                # Calculate torque
                # 计算扭矩
                qd_target = np.array(qd_target)
                qd_slide = np.array(qd_slide)
                tau_slide = Kp * (q_target - q_slide) + Kd * (qd_target - qd_slide)
                self.data.ctrl[15] = tau_slide                
            else :
                self.data.ctrl[15] = 0
        
        elif self.obstacle_mode == 'static':
            
            
            action = np.clip(action, -1, 1)  # 限制到[-1,1]
            qpos_target = self.action_to_qpos(action)
            qvel_target = 0.0
        
            # PD 控制扭矩
            torque = self.Kp * (qpos_target - qpos_now) + self.Kd * (qvel_target - qvel_now)
            self.data.ctrl[actutator_ids] = torque           
        
        else:
            # Only gravity acts, and the output torque is zero / 只有重力作用，输出扭矩皆为 0
            self.data.ctrl[actutator_ids] = 0                                    
                
        # =======================
        # Simulation one step / 仿真一步
        # ======================= 
        mujoco.mj_step(self.model, self.data)
        self.time += self.dt
        self.episode_step += 1

        # =======================
        # Render mode switching
        # =======================
        if self.render_mode == "human":
            self.render()
        elif self.render_mode == "rgb_array":
            frame = self.render()
            info["frame"] = frame
    
        # =======================
        # If the viewer is enabled, synchronize the screen / 如果开启了 viewer ，则同步画面
        # =======================
        if self.viewer is not None:
            self.viewer.sync()
        
        # =======================
        # Constructing observation space / 构造观测空间
        # ======================= 
        obs = np.concatenate([self.data.qpos[:8], 
                              self.data.qvel[:8], 
                              self.data.qpos[12:20]])

        # =======================
        # Constructing reward function / 构造奖励函数
        # =======================
        #1. Long distance punishment/close range reward
        #(Encourage moving towards the target)
        # 1、远距离惩罚/近距离奖励
        #（鼓励向目标移动）        
        agent_ee_body_id = self.model.body("tip8_agent").id
        agent_ee_pos = self.data.xpos[agent_ee_body_id]
        obstacle_base_body_id = self.model.body("base_link_obstacle").id
        target_pos =  self.data.xpos[obstacle_base_body_id]
        dist_to_target = np.linalg.norm(agent_ee_pos - target_pos)
        R_dist = dist_to_target**2
        w_dist = -1.0
        
        # 2、Reward for Success and Punishment for Failure
        # 2、成功奖励与失败惩罚
        ccp = self._check_collision_point ()
        R_success =  1.0 if ccp == 2 or ccp == 3 else 0.0
        w_success = 10000.0
        R_failure =  1.0 if ccp == 1 or ccp == 3 else 0.0
        w_failure = -10000.0       
        
        # 3. Collision rewards or punishments
        # (It is currently unclear whether collisions should be punished or rewarded, so w_collision is set to 0.0)
        # 3、碰撞奖励或惩罚
        # （暂时不清楚是否应该对碰撞进行惩罚或奖励，因此将 w_collision 设置为 0.0）
        ccnp = self._check_collision_no_point()
        R_collision = 1.0 if ccnp else 0.0
        w_collision = 0.0
        
        # 4、动作平滑度惩罚 或 控制开销惩罚
        #（鼓励更平滑的路径）
        # 4.Punishment for smoothness of actions
        # (Encourage smoother paths)
        R_action = np.sum(np.square(action))
        w_action = -0.001
        
        # 5. Time step punishment 
        # (encouraging efficiency, exponential growth to avoid small rewards drowning out the final reward)
        # 5、时间步惩罚
        # （鼓励效率，指数增长避免小奖励淹没最终奖励）
        R_step = 1
        w_step = -0.001
        
        R_separate = np.array([
            w_dist * R_dist,
            w_success * R_success,
            w_failure * R_failure,
            w_collision * R_collision,
            w_action * R_action,
            w_step * R_step
        ])
        
        # Calculate the reward function value / 计算奖励函数值
        reward = w_dist * R_dist + w_success * R_success + w_failure * R_failure +\
                 w_collision * R_collision + w_action * R_action + w_step * R_step

        # =======================
        # End flag / 结束标志
        # ======================= 
        
        terminated = self._check_collision_point() != 0
        truncated = self.episode_step >= self.max_episode_steps
        info = {
            "R_separate": R_separate
            }
        
        # =======================
        # Return / 返回
        # =======================
        return obs, reward, terminated, truncated, info
    
    
    # =======================
    # Render mode switching / 渲染环境切换
    # ======================= 
    def render(self):
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == 'rgb_array':
            # If necessary, implement off screen rendering / 如果需要，实现离屏渲染
            pass
    
    # =======================sdddd
    # Reset / 重置
    # =======================   
    def reset(self, seed=None, options=None):  
        super().reset(seed=seed)        
        # Reset simulation / 重置模拟状态
        mujoco.mj_resetData(self.model, self.data) 
        self.time = 0.0
        self.episode_step = 0
        # Constructing observation space / 构造观测空间
        obs = np.concatenate([self.data.qpos[:8], self.data.qvel[:8], self.data.qpos[12:20]]) 
        info = {} 
        # Return / 返回
        return obs, info  

    # =======================
    # Release resources / 释放资源
    # ======================= 
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            print("🛑 Close MuJoCo viewer / 关闭 MuJoCo viewer")
    
    
