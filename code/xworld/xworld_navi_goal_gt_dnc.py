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

    def gen_sample_seq(self, state):
        """
        generate path sequence to abstract and store in memory
        Return:
            state_seq: (seq_len, 4*3, 3, num_classes+2)
        """
        width, height = state.xmap.dim['width'], state.xmap.dim['height']
        num_classes = len(state.xmap.item_class_id)
        full_onehot_state = numpy.zeros((height+2, width+2, num_classes+2), dtype=bool)
        full_onehot_state[:,:,-1] = True
        full_onehot_state[1:-1, 1:-1, :] = state.origin_onehot_state
        #### remove agent
        full_onehot_state[state.agent_location[1]+1,state.agent_location[0]+1,18] = False
        full_onehot_state[state.agent_location[1]+1,state.agent_location[0]+1,0] = True
        state_seq = list()
        ngoal = 0
        for location, item_id in state.xmap.item_location_map.items():
            if state.xmap.items[item_id[0]].item_type == 'goal':
                ngoal += 1
        succeed_flag = False
        while not succeed_flag:
            tgt_loc = state.xmap.items[numpy.random.randint(ngoal)].location
            src_loc = numpy.array([numpy.random.randint(width), numpy.random.randint(height)])
            if (src_loc==tgt_loc).all() or numpy.abs(src_loc-tgt_loc).sum()<8:
                continue
            if state.origin_inner_state[src_loc[1], src_loc[0]]==0:
                succeed_flag, path_loc, path_ori = self.compute_shortest_path_single_src(state, src_loc, tgt_loc)

        for j, loc in enumerate(path_loc):
            state_ini = full_onehot_state[loc[1]:loc[1]+3, loc[0]:loc[0]+3, :]
            onehot_state = copy.deepcopy(numpy.rot90(state_ini, path_ori[j]))
            inner_state = numpy.argmax(onehot_state, axis=2)
            ## block view
            onehot_state[2,inner_state[1,:]==17,:] = False
            onehot_state[2,inner_state[1,:]==17,num_classes] = True
            state_seq.append(numpy.concatenate([numpy.rot90(onehot_state, 0), numpy.rot90(onehot_state, 1), 
                numpy.rot90(onehot_state, 2), numpy.rot90(onehot_state, 3)], 0)[None])

        state_seq = numpy.concatenate(state_seq, axis=0)
        return state_seq, path_loc, path_ori

    def gen_sample_long_seq(self, state, nsubgoal):
        """
        a sample long sequence goes to different places
        generate path sequence to abstract and store in memory
        Return:
            state_seq: (seq_len, 4*3, 3, num_classes+2)
        """
        width, height = state.xmap.dim['width'], state.xmap.dim['height']
        num_classes = len(state.xmap.item_class_id)
        full_onehot_state = numpy.zeros((height+2, width+2, num_classes+2), dtype=bool)
        full_onehot_state[:,:,-1] = True
        full_onehot_state[1:-1, 1:-1, :] = state.origin_onehot_state
        #### remove agent
        full_onehot_state[state.agent_location[1]+1,state.agent_location[0]+1,18] = False
        full_onehot_state[state.agent_location[1]+1,state.agent_location[0]+1,0] = True
        state_seq = list()
        path_loc_list = list()
        path_ori_list = list()
        flag_list = list()
        ngoal = 0
        for location, item_id in state.xmap.item_location_map.items():
            if state.xmap.items[item_id[0]].item_type == 'goal':
                ngoal += 1
        nsucceed = 0
        restart = True
        while nsucceed<nsubgoal:
            if restart:
                restart = False
                succeed_flag = False
                while not succeed_flag:
                    tgt_loc = numpy.array([numpy.random.randint(width), numpy.random.randint(height)])
                    if state.origin_inner_state[tgt_loc[1], tgt_loc[0]]!=17:
                        succeed_flag = True
            succeed_flag = False
            src_loc = copy.deepcopy(tgt_loc)
            fail_count = 0
            while not succeed_flag:
                if fail_count>10:
                    restart=True
                    if len(flag_list)>0:
                        flag_list[-1][-1] = 1
                    break
                tgt_loc = numpy.array([numpy.random.randint(width), numpy.random.randint(height)])
                if (src_loc==tgt_loc).all():
                    continue
                if state.origin_inner_state[tgt_loc[1], tgt_loc[0]]!=17:
                    succeed_flag, path_loc, path_ori = self.compute_shortest_path_single_src(state, src_loc, tgt_loc)
                fail_count += (1-succeed_flag)
            if succeed_flag:
                path_loc_list += path_loc[:-1]
                path_ori_list += path_ori[:-1]
                flag_list.append(numpy.zeros(len(path_loc)-1))
                nsucceed += 1
        flag_list = numpy.concatenate(flag_list, 0)
        flag_list[-1] = 1

        for j, loc in enumerate(path_loc_list):
            state_ini = full_onehot_state[loc[1]:loc[1]+3, loc[0]:loc[0]+3, :]
            onehot_state = copy.deepcopy(numpy.rot90(state_ini, path_ori_list[j]))
            inner_state = numpy.argmax(onehot_state, axis=2)
            ## block view
            onehot_state[2,inner_state[1,:]==17,:] = False
            onehot_state[2,inner_state[1,:]==17,num_classes] = True
            state_seq.append(numpy.concatenate([numpy.rot90(onehot_state, 0), numpy.rot90(onehot_state, 1), 
                numpy.rot90(onehot_state, 2), numpy.rot90(onehot_state, 3)], 0)[None])

        state_seq = numpy.concatenate(state_seq, axis=0)
        return state_seq, path_loc_list, path_ori_list, flag_list

    def gen_sample_query(self, state, loc_pool=None):
        """
        generate a sample query
        Return:
            src_inputs: (4*4, nrow, ncol, num_classes+2)
            tgt_inputs: (4*4, nrow, ncol, num_classes+2)
        """
        width, height = state.xmap.dim['width'], state.xmap.dim['height']
        side_radius = self.args.visible_radius_unit_side
        front_radius = self.args.visible_radius_unit_front
        radius = max(side_radius, front_radius)
        num_classes = len(state.xmap.item_class_id)
        full_onehot_state = numpy.zeros((height+2*radius, width+2*radius, num_classes+2), dtype=bool)
        full_onehot_state[:,:,-1] = True
        full_onehot_state[radius:radius+height, radius:radius+width, :] = state.origin_onehot_state
        #### remove agent
        full_onehot_state[state.agent_location[1]+radius,state.agent_location[0]+radius,18] = False
        full_onehot_state[state.agent_location[1]+radius,state.agent_location[0]+radius,0] = True
        full_inner_state = numpy.argmax(full_onehot_state, 2)
        ngoal = 0
        for location, item_id in state.xmap.item_location_map.items():
            if state.xmap.items[item_id[0]].item_type == 'goal':
                ngoal += 1
        goal_dist_list = list()
        for i in range(ngoal):
            for j in range(i+1, ngoal):
                goal_dist_list.append(numpy.abs(state.xmap.items[i].location-state.xmap.items[j].location).sum())
        goal_diameter = sum(goal_dist_list)/len(goal_dist_list)
        succeed_flag = False
        while not succeed_flag:
            if loc_pool is not None:
                nloc = len(loc_pool)
                #tgt_loc = loc_pool[numpy.random.randint(nloc)]
                tgt_loc = state.xmap.items[numpy.random.randint(ngoal)].location
                src_loc = loc_pool[numpy.random.randint(nloc)]
            else:
                tgt_loc = state.xmap.items[numpy.random.randint(ngoal)].location
                src_loc = numpy.array([numpy.random.randint(width), numpy.random.randint(height)])
            if (src_loc==tgt_loc).all() or numpy.abs(tgt_loc-src_loc).sum()<goal_diameter:
                continue
            if state.origin_inner_state[src_loc[1], src_loc[0]]!=17:
                succeed_flag, path_loc, path_ori = self.compute_shortest_path_single_src(state, src_loc, tgt_loc)

        query_inputs = list()
        for ii in range(2):
            query_inputs.append(list())
            if ii==0:
                cur_loc = src_loc
            else:
                cur_loc = tgt_loc
            for j in range(4):
                rotflag = j
                if rotflag==0:
                    start_x = radius + cur_loc[0]
                    start_y = radius + cur_loc[1]
                elif rotflag==1:
                    start_x = radius + cur_loc[1]
                    start_y = radius + width - cur_loc[0]-1
                elif rotflag==2:
                    start_x = radius + width - cur_loc[0]-1
                    start_y = radius + height - cur_loc[1]-1
                elif rotflag==3:
                    start_x = radius + height - cur_loc[1]-1
                    start_y = radius + cur_loc[0]
                cur_inner_state = copy.deepcopy(numpy.rot90(full_inner_state, rotflag)[start_y:start_y+front_radius+1, start_x-side_radius:start_x+side_radius+1])
                cur_onehot_state = copy.deepcopy(numpy.rot90(full_onehot_state, rotflag)[start_y:start_y+front_radius+1, start_x-side_radius:start_x+side_radius+1, :])
                # block views after the wall
                for i in range(2*side_radius+1):
                    for j in range(1,front_radius+1):
                        if cur_inner_state[j,i]==17:
                            cur_inner_state[j+1:,i] = num_classes
                            cur_onehot_state[j+1:, i, :] = False
                            cur_onehot_state[j+1:, i, num_classes] = True
                query_inputs[ii].append(numpy.rot90(cur_onehot_state,0)[None])
                query_inputs[ii].append(numpy.rot90(cur_onehot_state,1)[None])
                query_inputs[ii].append(numpy.rot90(cur_onehot_state,2)[None])
                query_inputs[ii].append(numpy.rot90(cur_onehot_state,3)[None])

        src_inputs = numpy.concatenate(query_inputs[0], 0)
        tgt_inputs = numpy.concatenate(query_inputs[1], 0)

        goal_obs_onehot_state = copy.deepcopy(full_onehot_state[radius+tgt_loc[1]-1:radius+tgt_loc[1]+2, radius+tgt_loc[0]-1:radius+tgt_loc[0]+2, :])
        goal_obs_inner_state = numpy.argmax(goal_obs_onehot_state, axis=2)
        ## block view
        goal_obs_onehot_state[2,goal_obs_inner_state[1,:]==17,:] = False
        goal_obs_onehot_state[2,goal_obs_inner_state[1,:]==17,num_classes] = True
        return src_inputs, tgt_inputs, src_loc, tgt_loc, goal_obs_onehot_state

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
