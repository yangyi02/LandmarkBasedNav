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
        self.goal_locations = []
        self.wrong_goal_locations = []
        self.rewards['navi_goal'] = 0.0
        self.israndom_goal = True
        self.goal_id = 0
        self.goal_obs_onehot_state = []

    def reset_command(self, state):
        """
        The command should contain goal name so the agent knows where to go
        It is possible there are multiple same goals in the map
        The function will return all the goal locations
        """
        goal_list = []
        for location, item_id in state.xmap.item_location_map.items():
            if state.xmap.items[item_id[0]].item_type == 'goal':
                goal_list.append(item_id[0])
        assert len(goal_list) > 0, "error: at least one goal is needed for this task"
        if self.israndom_goal:
            goal_item_id = goal_list[random.randint(0, len(goal_list)-1)]
        else:
            goal_item_id = self.goal_id
        goal_class_name = state.xmap.items[goal_item_id].class_name

        num_classes = len(state.xmap.item_class_id)
        self.goal_obs_onehot_state = numpy.full(num_classes+1, 0, dtype=bool)
        self.goal_obs_onehot_state[state.xmap.items[goal_item_id].class_id] = True

        if self.args.single_word:
            self.command = goal_class_name
        else:
            self.command = 'go to ' + goal_class_name
        self.goal_locations = []
        self.wrong_goal_locations = []
        for goal_item_id in goal_list:
            item = state.xmap.items[goal_item_id]
            if item.class_name == goal_class_name:
                self.goal_locations.append(item.location)
            else:
                self.wrong_goal_locations.append(item.location)

    def update_reward(self, agent, state, action, next_state, num_step):
        self.update_navi_reward(agent, state, action, next_state, num_step)
        self.update_step_reward(agent, state, action, next_state, num_step)
        self.update_out_border_reward(agent, state, action, next_state, num_step)
        self.update_knock_block_reward(agent, state, action, next_state, num_step)

    def update_knock_block_reward(self, agent, state, action, next_state, num_step):
        self.rewards['knock_block'] = 0.0
        
        agent_id = state.xmap.item_name_map[agent.name]
        velocity = agent.velocity[action]
        next_location = state.xmap.items[agent_id].get_next_location(velocity)
        if tuple(next_location) in state.xmap.item_location_map:
            for next_item_id in state.xmap.item_location_map[tuple(next_location)]:
                next_item_type = state.xmap.items[next_item_id].item_type
                if next_item_type == 'block':
                    self.rewards['knock_block'] = -0.1
                    break
        for wrong_goal_location in self.wrong_goal_locations:
            if (next_location == wrong_goal_location).all():
                self.rewards['knock_block'] = -0.1
                break

    def update_navi_reward(self, agent, state, action, next_state, num_step):
        """
        The agent get positive reward when navigation reach goal
        """
        agent_id = state.xmap.item_name_map[agent.name]
        velocity = agent.velocity[action]
        next_location = state.xmap.items[agent_id].get_next_location(velocity)
        if tuple(next_location) in state.xmap.item_location_map:
            for next_item_id in state.xmap.item_location_map[tuple(next_location)]:
                next_item_type = state.xmap.items[next_item_id].item_type
                if next_item_type == 'river':
                    self.rewards['navi_goal'] = -1.0
                    self.done = True
                    return
        #agent_id = next_state.xmap.item_name_map[agent.name]
        #agent_location = next_state.xmap.items[agent_id].location
        for goal_location in self.goal_locations:
            if (next_location == goal_location).all():
                self.rewards['navi_goal'] = 1.0
                self.done = True
                return
        #for i in range(len(self.wrong_goal_locations)):
        #    if (agent_location == self.wrong_goal_locations[i]).all():
        #        self.rewards['navi_goal'] = -1.0
        #        self.wrong_goal_locations.pop(i)
        #        return