import random
from .xworld import XWorld
from .xworld_teacher import XWorldTeacher
import logging
import numpy
from scipy import ndimage
from collections import deque
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)
import copy


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
        self.goal_location = []
        self.israndom_goal = args.israndom_goal
        self.goal_id = args.goal_id

    def reset_command(self, state):
        """
        The command should contain goal name so the agent knows where to go
        It is possible there are multiple same goals in the map
        The function will return all the goal locations
        """
        if state.is_render_image:
            self.update_goal_obs_image(state)
        goal_list = []
        for location, item_id in state.xmap.item_location_map.items():
            if state.xmap.items[item_id[0]].item_type == 'goal':
                goal_list.append(item_id[0])
        if self.israndom_goal:
            goal_item_id = goal_list[random.randint(0, len(goal_list)-1)]
        else:
            goal_item_id = goal_list[self.goal_id]
        self.goal_class_name = state.xmap.items[goal_item_id].class_name
        self.goal_location = state.xmap.items[goal_item_id].location
        self.compute_shortest_path(state)

    def update_goal_obs_image(self, state):
        width, height = state.xmap.dim['width'], state.xmap.dim['height']
        side_radius = min(self.args.visible_radius_unit_side, max(width - 1, height - 1))
        block_size = state.image_block_size
        self.goal_obs_image = state.image[:3*block_size, (side_radius-1)*block_size:(side_radius+2)*block_size, :]

    def compute_shortest_path(self, state):
        width, height = state.xmap.dim['width'], state.xmap.dim['height']
        # only empty space is walkable
        visit_map = numpy.logical_not(numpy.logical_or(state.origin_inner_state==0, state.origin_inner_state==18))
        # action map -1block 0_ 1<- 2^ 3->
        action_map = -numpy.ones([height, width]).astype('int32')
        dist_map = numpy.inf*numpy.ones([height, width])
        dist_map[self.goal_location[1], self.goal_location[0]] = 0
        d = deque()
        d.append(self.goal_location)
        while d:
            cur_loc = d.popleft()
            visit_map[cur_loc[1], cur_loc[0]] = True
            offsets = [numpy.array([0, -1]), numpy.array([1, 0]), numpy.array([0, 1]), numpy.array([-1, 0])]
            actions = numpy.array([0, 1, 2, 3], 'int32')
            for i in range(4):
                new_loc = cur_loc+offsets[i]
                if (new_loc>=0).sum()==2 and (new_loc<[width, height]).sum()==2:
                    if not visit_map[new_loc[1], new_loc[0]]:
                        cur_dist = dist_map[cur_loc[1], cur_loc[0]]+1+(action_map[cur_loc[1], cur_loc[0]]!=actions[i])
                        if cur_dist<dist_map[new_loc[1], new_loc[0]]:
                            dist_map[new_loc[1], new_loc[0]] = cur_dist
                            action_map[new_loc[1], new_loc[0]] = actions[i]
                        d.append(new_loc)
        self.action_map = action_map
        self.dist_map = dist_map

    def compute_shortest_path_single_src(self, state, src_loc, tgt_loc):
        width, height = state.xmap.dim['width'], state.xmap.dim['height']
        # only empty space is walkable
        visit_map = numpy.logical_not(numpy.logical_or(state.origin_inner_state==0, state.origin_inner_state==18))
        visit_map[src_loc[1], src_loc[0]] = False
        # action map -1block 0_ 1<- 2^ 3->
        action_map = -numpy.ones([height, width]).astype('int32')
        dist_map = numpy.inf*numpy.ones([height, width])
        dist_map[tgt_loc[1], tgt_loc[0]] = 0
        d = deque()
        d.append(tgt_loc)
        orientation_map = [numpy.array([0, 1]), numpy.array([-1, 0]), numpy.array([0, -1]), numpy.array([1, 0])]
        while (not visit_map[src_loc[1], src_loc[0]]) and d:
            cur_loc = d.popleft()
            visit_map[cur_loc[1], cur_loc[0]] = True
            offsets = [numpy.array([0, -1]), numpy.array([1, 0]), numpy.array([0, 1]), numpy.array([-1, 0])]
            actions = numpy.array([0, 1, 2, 3], 'int32')
            for i in range(4):
                new_loc = cur_loc+offsets[i]
                if (new_loc>=0).sum()==2 and (new_loc<[width, height]).sum()==2:
                    if not visit_map[new_loc[1], new_loc[0]]:
                        cur_dist = dist_map[cur_loc[1], cur_loc[0]]+1+(action_map[cur_loc[1], cur_loc[0]]!=actions[i])
                        if cur_dist<dist_map[new_loc[1], new_loc[0]]:
                            dist_map[new_loc[1], new_loc[0]] = cur_dist
                            action_map[new_loc[1], new_loc[0]] = actions[i]
                        d.append(new_loc)
        path_loc = []
        path_ori = []
        path_loc.append(copy.deepcopy(src_loc))
        path_ori.append(copy.deepcopy(action_map[src_loc[1], src_loc[0]]))
        succeed_flag = True
        if visit_map[src_loc[1], src_loc[0]]:
            cur_loc = copy.deepcopy(src_loc)
            while (cur_loc==tgt_loc).sum()<2:
                cur_loc += orientation_map[action_map[cur_loc[1], cur_loc[0]]]
                path_ori.append(copy.deepcopy(action_map[cur_loc[1], cur_loc[0]]))
                path_loc.append(copy.deepcopy(cur_loc))
            path_ori[-1] = path_ori[-2]
        else:
            succeed_flag = False
        return succeed_flag, path_loc, path_ori

    def update_reward(self, agent, state, action, next_state, num_step):
        self.update_navi_reward(agent, state, action, next_state, num_step)
        self.rewards['step'] = 0.0
        self.rewards['out_border'] = 0.0
        self.rewards['knock_block'] = 0.0

    def update_navi_reward(self, agent, state, action, next_state, num_step):
        """
        The agent get positive reward when navigation reach goal
        """
        
        agent_id = state.xmap.item_name_map[agent.name]
        velocity = agent.velocity[action]
        next_location = state.xmap.items[agent_id].get_next_location(velocity)
        if (next_location == self.goal_location).all():
            self.rewards['navi_goal'] = 10.0
            self.done = True
            return