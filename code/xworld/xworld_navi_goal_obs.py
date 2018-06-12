import random
from .xworld import XWorld
from .xworld_teacher import XWorldTeacher
import logging
import numpy
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class XWorldNaviGoal(XWorld):
    """
    XWorld interface for xworld robot learning
    """
    def __init__(self, args):
        super().__init__(args)
        self.teacher = XWorldTeacherNaviGoal(args)


class XWorldTeacherNaviGoal(XWorldTeacher):
    """
    XWorld reward for navigation goal task
    """
    def __init__(self, args):
        super().__init__(args)
        self.rewards['navi_goal'] = 0.0
        self.goal_obs_image = []
        self.goal_obs_inner_state = []
        self.goal_obs_onehot_state = []

    def reset_command(self, state):
        """
        The command should contain goal name so the agent knows where to go
        It is possible there are multiple same goals in the map
        The function will return all the goal locations
        """

        width, height = state.xmap.dim['width'], state.xmap.dim['height']
        num_classes = len(state.xmap.item_class_id)
        side_radius = self.args.visible_radius_unit_side #min(self.args.visible_radius_unit_side, max(width - 1, height - 1))
        front_radius = self.args.visible_radius_unit_front #min(self.args.visible_radius_unit_front, max(width - 1, height - 1))
        block_size = state.image_block_size
        radius = max(side_radius, front_radius)
        self.goal_obs_inner_state = numpy.full((height+2*radius, width+2*radius), -1, dtype=int)
        self.goal_obs_onehot_state = numpy.full((height+2*radius, width+2*radius, num_classes+1), 0, dtype=bool)
        if state.is_render_image:
            self.goal_obs_image = numpy.full(((height+2*radius) * block_size, (width+2*radius) * block_size, 3), 0.5)
            self.goal_obs_image[radius*block_size:(radius+height)*block_size, radius*block_size:(radius+width)*block_size, :] = state.origin_image
        rotflag = numpy.random.randint(4)
        while 1:
            self.goal_location = numpy.array([numpy.random.randint(width),numpy.random.randint(height)])
            if state.origin_inner_state[self.goal_location[1], self.goal_location[0]]==0:
                break
        if rotflag==0:
            start_x = radius + self.goal_location[0]
            start_y = radius + self.goal_location[1]
        elif rotflag==1:
            start_x = radius + self.goal_location[1]
            start_y = radius + width - self.goal_location[0]-1
        elif rotflag==2:
            start_x = radius + width - self.goal_location[0]-1
            start_y = radius + height - self.goal_location[1]-1
        elif rotflag==3:
            start_x = radius + height - self.goal_location[1]-1
            start_y = radius + self.goal_location[0]
        self.goal_obs_inner_state[radius:radius+height, radius:radius+width] = state.origin_inner_state
        self.goal_obs_onehot_state[radius:radius+height, radius:radius+width, :] = state.origin_onehot_state
        self.goal_obs_inner_state = numpy.rot90(self.goal_obs_inner_state, rotflag)[start_y:start_y+front_radius+1, start_x-side_radius:start_x+side_radius+1]
        self.goal_obs_onehot_state = numpy.rot90(self.goal_obs_onehot_state, rotflag)[start_y:start_y+front_radius+1, start_x-side_radius:start_x+side_radius+1, :]
        if state.is_render_image:
            self.goal_obs_image = numpy.rot90(self.goal_obs_image, rotflag)[(start_y)*block_size:(start_y+front_radius+1)*block_size, (start_x-side_radius)*block_size:(start_x+side_radius+1)*block_size, :]
        # block views after the wall
        for i in range(2*side_radius+1):
            for j in range(1,front_radius+1):
                if self.goal_obs_inner_state[j,i]==17:
                    self.goal_obs_inner_state[j+1:,i] = num_classes
                    self.goal_obs_onehot_state[j+1:, i, :] = False
                    self.goal_obs_onehot_state[j+1:, i, num_classes] = True
                    if state.is_render_image:
                        self.goal_obs_image[(j+1)*block_size:,i*block_size:(i+1)*block_size,:] = 0.2
                    break
        for i in range(2*side_radius+1):
            for j in range(front_radius+1):
                if self.goal_obs_inner_state[j, i]==18:
                    self.goal_obs_inner_state[j, i] = 0
                    self.goal_obs_onehot_state[j, i, 18] = False
                    self.goal_obs_onehot_state[j, i, 0] = True
                    if state.is_render_image:
                        self.goal_obs_image[j*block_size:(j+1)*block_size,i*block_size:(i+1)*block_size,:] = 1
        self.goal_obs_inner_state[0, side_radius] = 18
        self.goal_obs_onehot_state[0, side_radius, 18] = True
        self.goal_obs_onehot_state[0, side_radius, 0] = False
        #self.goal_obs_image[:block_size,side_radius*block_size:(side_radius+1)*block_size,:] = 0.5

    def update_reward(self, agent, state, action, next_state, num_step):
        self.update_navi_reward(agent, state, action, next_state, num_step)
        self.update_step_reward(agent, state, action, next_state, num_step)
        self.update_out_border_reward(agent, state, action, next_state, num_step)
        self.update_knock_block_reward(agent, state, action, next_state, num_step)

    def update_navi_reward(self, agent, state, action, next_state, num_step):
        """
        The agent get positive reward when navigation reach goal
        """
        if (state.inner_state==self.goal_obs_inner_state).reshape(-1).all():
            self.rewards['navi_goal'] = 1.0
            self.done = True
            return