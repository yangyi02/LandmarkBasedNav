import random
from .xworld import XWorld
from .xworld_teacher import XWorldTeacher
import logging
import numpy
from scipy import ndimage
import copy
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
        self.goal_location = []
        self.israndom_goal = False
        self.goal_id = 0
        self.goal_finish = False

    def reset_command(self, state):
        """
        The command should contain goal name so the agent knows where to go
        It is possible there are multiple same goals in the map
        The function will return all the goal locations
        """

        '''
        goal_list = []
        for location, item_id in state.xmap.item_location_map.items():
            if state.xmap.items[item_id[0]].item_type == 'goal':
                goal_list.append(item_id[0])
        if self.israndom_goal:
            goal_item_id = goal_list[random.randint(0, len(goal_list)-1)]
        else:
            goal_item_id = self.goal_id
        goal_class_name = state.xmap.items[goal_item_id].class_name
        self.goal_locations = []
        for goal_item_id in goal_list:
            item = state.xmap.items[goal_item_id]
            if item.class_name == goal_class_name:
                self.goal_locations.append(item.location)
        '''
        self.goal_finish = False
        width, height = state.xmap.dim['width'], state.xmap.dim['height']
        num_classes = len(state.xmap.item_class_id)
        side_radius = self.args.visible_radius_unit_side #min(self.args.visible_radius_unit_side, max(width - 1, height - 1))
        front_radius = self.args.visible_radius_unit_front #min(self.args.visible_radius_unit_front, max(width - 1, height - 1))
        block_size = state.image_block_size
        radius = max(side_radius, front_radius)
        while 1:
            self.goal_obs_inner_state = numpy.full((height+2*radius, width+2*radius), num_classes+1, dtype=int)
            self.goal_obs_onehot_state = numpy.full((height+2*radius, width+2*radius, num_classes+2), 0, dtype=bool)
            self.goal_obs_onehot_state[:,:,-1] = True
            if state.is_render_image:
                self.goal_obs_image = numpy.full(((height+2*radius) * block_size, (width+2*radius) * block_size, 3), 0.5)
                self.goal_obs_image[radius*block_size:(radius+height)*block_size, radius*block_size:(radius+width)*block_size, :] = state.origin_image

            rotflag = numpy.random.randint(4)
            self.rotflag = rotflag
            if rotflag==0:
                orientation = numpy.array([0, 1])
            elif rotflag==1:
                orientation = numpy.array([-1, 0])
            elif rotflag==2:
                orientation = numpy.array([0, -1])
            elif rotflag==3:
                orientation = numpy.array([1, 0])
            while 1:  
                self.goal_location = numpy.array([numpy.random.randint(width),numpy.random.randint(height)])
                if (state.origin_inner_state[self.goal_location[1], self.goal_location[0]]==0):
                    break
                    #if numpy.sum((self.goal_location+orientation)>=0)==2 and numpy.sum((self.goal_location+orientation)<numpy.array([width, height]))==2:
                    #    if not (state.origin_inner_state[self.goal_location[1]+orientation[1], self.goal_location[0]+orientation[0]] in [17,22,23]):
                    #        break
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
                        self.goal_obs_onehot_state[j+1:, i, :] = True # missing in goal could be everything
                        #self.goal_obs_onehot_state[j+1:, i, :] = False
                        #self.goal_obs_onehot_state[j+1:, i, num_classes] = True
                        if state.is_render_image:
                            self.goal_obs_image[(j+1)*block_size:,i*block_size:(i+1)*block_size,:] = 0.2
                        break
            tmp = self.goal_obs_inner_state[:3,side_radius-1:side_radius+2].reshape(-1)
            if numpy.logical_and(tmp<=17, tmp>0).sum()>0:
                break

        #### remove agent ####
        for i in range(2*side_radius+1):
            for j in range(front_radius+1):
                if self.goal_obs_inner_state[j, i]==18:
                    self.goal_obs_inner_state[j, i] = 0
                    self.goal_obs_onehot_state[j, i, 18] = False
                    self.goal_obs_onehot_state[j, i, 0] = True
                    if state.is_render_image:
                        self.goal_obs_image[j*block_size:(j+1)*block_size,i*block_size:(i+1)*block_size,:] = 1
        #self.goal_obs_inner_state[0, side_radius] = 18
        #self.goal_obs_onehot_state[0, side_radius, 18] = True
        #self.goal_obs_onehot_state[0, side_radius, 0] = False
        #self.goal_obs_image[:block_size,side_radius*block_size:(side_radius+1)*block_size,:] = 0.5
        #### crop to get 3x3 goal ####
        self.goal_obs_inner_state = self.goal_obs_inner_state[:3,side_radius-1:side_radius+2]
        self.goal_obs_onehot_state = self.goal_obs_onehot_state[:3,side_radius-1:side_radius+2,:]
        if state.is_render_image:
            self.goal_obs_image = self.goal_obs_image[0:3*block_size,(side_radius-1)*block_size:(side_radius+2)*block_size,:]
        #### treat missing observation as correct observation ####
        self.goal_obs_onehot_state[:,:,num_classes] = True
        #### rotate ####
        self.goal_obs_onehot_state = numpy.expand_dims(self.goal_obs_onehot_state, 3)
        self.goal_obs_onehot_state = numpy.concatenate([numpy.rot90(self.goal_obs_onehot_state, 0), numpy.rot90(self.goal_obs_onehot_state, 1),
            numpy.rot90(self.goal_obs_onehot_state, 2), numpy.rot90(self.goal_obs_onehot_state, 3)], 3)
        self.goal_obs_inner_state = numpy.expand_dims(self.goal_obs_inner_state, 2)
        self.goal_obs_inner_state = numpy.concatenate([numpy.rot90(self.goal_obs_inner_state, 0), numpy.rot90(self.goal_obs_inner_state, 1),
            numpy.rot90(self.goal_obs_inner_state, 2), numpy.rot90(self.goal_obs_inner_state, 3)], 2)

        x = numpy.full((height+2, width+2, num_classes+2), 0, dtype=bool)
        x[:,:,-1] = True
        x[1:-1, 1:-1, :] = state.origin_onehot_state
        assert(x[state.agent_location[1]+1, state.agent_location[0]+1, 18] == True)
        x[state.agent_location[1]+1, state.agent_location[0]+1, 18] = False
        x[state.agent_location[1]+1, state.agent_location[0]+1, 0] = True
        responses = []
        for i in range(4):
            responses.append(ndimage.correlate(x.astype('float32'), self.goal_obs_onehot_state[:,:,:,i], mode='constant', cval=0.0)[1:-1, 1:-1, [int((x.shape[2]+1)/2)]])
        self.goal_responses = numpy.amax(numpy.concatenate(responses, 2), 2)==9

        self.goal_location += orientation

    def update_goal_obs_image(self, state):
        width, height = state.xmap.dim['width'], state.xmap.dim['height']
        side_radius = min(self.args.visible_radius_unit_side, max(width - 1, height - 1))
        block_size = state.image_block_size
        self.goal_obs_image = state.image[:3*block_size, (side_radius-1)*block_size:(side_radius+2)*block_size, :]

    def set_goal(self, goal_obs_onehot_state, goal_location):
        self.goal_obs_onehot_state = copy.deepcopy(goal_obs_onehot_state)
        self.goal_location = goal_location
        self.goal_finish = False
        self.done = False

    def update_reward(self, agent, state, action, next_state, num_step):
        self.update_navi_reward(agent, state, action, next_state, num_step)
        self.update_step_reward(agent, state, action, next_state, num_step)
        self.update_out_border_reward(agent, state, action, next_state, num_step)
        self.update_knock_block_reward(agent, state, action, next_state, num_step)

    def update_navi_reward(self, agent, state, action, next_state, num_step):
        """
        The agent get positive reward when navigation reach goal
        """
        
        """
        ##### for training ######
        agent_id = state.xmap.item_name_map[agent.name]
        velocity = agent.velocity[action]
        next_location = state.xmap.items[agent_id].get_next_location(velocity)
        if (next_location>=0).sum()==2 and (next_location<self.goal_responses.shape).sum()==2:
            if self.goal_responses[next_location[1], next_location[0]]:
                self.rewards['navi_goal'] = 1.0
                self.done = True
                return
        """
        """
        side_radius = self.args.visible_radius_unit_side
        curr_goal_obs_onehot_state = state.onehot_state[:3,side_radius-1:side_radius+2,:]
        if ( ((numpy.multiply(curr_goal_obs_onehot_state, self.goal_obs_onehot_state[:,:,:,0]).sum()==9) or
            (numpy.multiply(curr_goal_obs_onehot_state, self.goal_obs_onehot_state[:,:,:,1]).sum()==9) or
            (numpy.multiply(curr_goal_obs_onehot_state, self.goal_obs_onehot_state[:,:,:,2]).sum()==9) or
            (numpy.multiply(curr_goal_obs_onehot_state, self.goal_obs_onehot_state[:,:,:,3]).sum()==9)) and
            action=='move_forward' ):
            self.rewards['navi_goal'] = 1.0
            self.done = True
            return
        """

        ##### for testing ######
        
        side_radius = self.args.visible_radius_unit_side
        curr_goal_obs_onehot_state = state.onehot_state[:3,side_radius-1:side_radius+2,:]
        if ( ((numpy.multiply(curr_goal_obs_onehot_state, self.goal_obs_onehot_state[:,:,:,0]).sum()==9) or
            (numpy.multiply(curr_goal_obs_onehot_state, self.goal_obs_onehot_state[:,:,:,1]).sum()==9) or
            (numpy.multiply(curr_goal_obs_onehot_state, self.goal_obs_onehot_state[:,:,:,2]).sum()==9) or
            (numpy.multiply(curr_goal_obs_onehot_state, self.goal_obs_onehot_state[:,:,:,3]).sum()==9)) and
            action=='move_forward' ):
            self.rewards['navi_goal'] = 0.0
            self.done = True

            agent_id = state.xmap.item_name_map[agent.name]
            velocity = agent.velocity[action]
            next_location = state.xmap.items[agent_id].get_next_location(velocity)
            if (next_location == self.goal_location).all():
                self.goal_finish = True
            return
        