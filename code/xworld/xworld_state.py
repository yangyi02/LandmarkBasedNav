import numpy
import copy
from . import xworld_map_dungeon
#from . import xworld_map
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class XWorldState(object):
    """
    XWorld state and reward for xworld robot learning
    """
    def __init__(self, args):
        """
        The xworld state contains all the information from the environment, including image and teacher's language as well as feedback reward from teacher
        """
        self.args = args
        self.image_block_size = args.image_block_size
        self.xmap = xworld_map_dungeon.XWorldMap(args.map_config, args.image_block_size)
        #self.xmap = xworld_map.XWorldMap(args.map_config, args.image_block_size)
        self.image = []
        self.inner_state = []
        self.onehot_state = []
        self.origin_image = []
        self.origin_inner_state = []
        self.origin_onehot_state = []
        self.plain_ground_image = []
        self.agent_location = []
        self.agent_orientation = []
        self.init_plain_ground_image()
        self.is_render_image = False

    def seed(self, seed=None):
        self.xmap.seed(seed)

    def init_plain_ground_image(self):
        width, height = self.xmap.dim['width'], self.xmap.dim['height']
        block_size = self.image_block_size
        self.plain_ground_image = numpy.full((height * block_size, width * block_size, 3), 1, dtype=float)
        # create dark line for visualizing grid
        for i in range(0, height * block_size, block_size):
            self.plain_ground_image[i, 0:width * block_size, :] = 0
        for i in range(block_size-1, height * block_size, block_size):
            self.plain_ground_image[i, 0:width * block_size, :] = 0
        for j in range(0, width * block_size, block_size):
            self.plain_ground_image[0:height * block_size, j, :] = 0
        for j in range(block_size-1, width * block_size, block_size):
            self.plain_ground_image[0:height * block_size, j, :] = 0

    def reset(self, agent):
        self.xmap.reset()
        self.init_plain_ground_image()
        self.construct_inner_state(agent)
        if self.is_render_image:
            self.construct_image(agent)

    def step(self, agent, action):
        self.xmap.step(agent, action)
        self.construct_inner_state(agent)
        if self.is_render_image:
            self.construct_image(agent)

    def copy_state(self, agent, src_state):
        self.xmap.copy_map(src_state.xmap)
        self.init_plain_ground_image()
        self.construct_inner_state(agent)
        if self.is_render_image:
            self.construct_image(agent)

    def teleport(self, agent, target_loc, target_ori):
        agent_id =self.xmap.item_name_map[agent.name]
        self.xmap.items[agent_id].location = target_loc
        self.xmap.items[agent_id].orientation = target_ori
        self.construct_inner_state(agent)
        if self.is_render_image:
            self.construct_image(agent)

    def construct_inner_state(self, agent):
        """
        Construct simple inner state image
        One can interpret as the abstract of full state image
        """
        width, height = self.xmap.dim['width'], self.xmap.dim['height']
        num_classes = len(self.xmap.item_class_id)
        self.origin_inner_state = numpy.zeros((height, width), dtype=int)
        self.origin_onehot_state = numpy.zeros((height, width, num_classes+2), dtype=bool) # last-1 dim is blocked unknown and last dim is out of boundary
        self.origin_onehot_state[:, :, 0] = True
        for item in self.xmap.items:
            if not item.is_removed:
                location = item.location
                self.origin_inner_state[location[1], location[0]] = item.class_id
                self.origin_onehot_state[location[1], location[0], item.class_id] = True
                self.origin_onehot_state[location[1], location[0], 0] = False
        agent_id = self.xmap.item_name_map[agent.name]
        self.agent_location = self.xmap.items[agent_id].location
        self.agent_orientation = self.xmap.items[agent_id].orientation
        if self.args.ego_centric:
            side_radius = self.args.visible_radius_unit_side #min(self.args.visible_radius_unit_side, max(width - 1, height - 1))
            front_radius = self.args.visible_radius_unit_front #min(self.args.visible_radius_unit_front, max(width - 1, height - 1))
            radius = max(side_radius, front_radius)
            self.inner_state = numpy.full((height+2*radius, width+2*radius), num_classes+1, dtype=int)
            self.onehot_state = numpy.full((height+2*radius, width+2*radius, num_classes+2), 0, dtype=bool)
            self.onehot_state[:, :, -1] = True # out of boundary
            rotflag = 0
            if self.agent_orientation[0]==0 and self.agent_orientation[1]==1:
                rotflag = 0
                start_x = radius + self.agent_location[0]
                start_y = radius + self.agent_location[1]
            elif self.agent_orientation[0]==-1 and self.agent_orientation[1]==0:
                rotflag = 1
                start_x = radius + self.agent_location[1]
                start_y = radius + width - self.agent_location[0]-1
            elif self.agent_orientation[0]==0 and self.agent_orientation[1]==-1:
                rotflag = 2
                start_x = radius + width - self.agent_location[0]-1
                start_y = radius + height - self.agent_location[1]-1
            elif self.agent_orientation[0]==1 and self.agent_orientation[1]==0:
                rotflag = 3
                start_x = radius + height - self.agent_location[1]-1
                start_y = radius + self.agent_location[0]
            self.inner_state[radius:radius+height, radius:radius+width] = copy.deepcopy(self.origin_inner_state)
            self.onehot_state[radius:radius+height, radius:radius+width, :] = copy.deepcopy(self.origin_onehot_state)
            self.inner_state = numpy.rot90(self.inner_state, rotflag)[start_y:start_y+front_radius+1, start_x-side_radius:start_x+side_radius+1]
            self.onehot_state = numpy.rot90(self.onehot_state, rotflag)[start_y:start_y+front_radius+1, start_x-side_radius:start_x+side_radius+1, :]

            # block views after the wall
            for i in range(2*side_radius+1):
                for j in range(1,front_radius+1):
                    if self.inner_state[j,i]==17:
                        self.inner_state[j+1:,i] = num_classes
                        self.onehot_state[j+1:, i, :] = False
                        self.onehot_state[j+1:, i, num_classes] = True
                        break
            # remove agent label
            self.inner_state[0, side_radius] = 0
            self.onehot_state[0, side_radius, 18] = False
            self.onehot_state[0, side_radius, 0] = True
        else:
            self.inner_state = self.origin_inner_state
            self.onehot_state = self.origin_onehot_state

    def construct_image(self, agent):
        """
        Construct the full state image
        """
        width, height = self.xmap.dim['width'], self.xmap.dim['height']
        block_size = self.image_block_size
        self.origin_image = numpy.copy(self.plain_ground_image)
        for item in self.xmap.items:
            if not item.is_removed:
                location = item.location
                start_x = location[0] * block_size
                start_y = location[1] * block_size
                end_x = start_x + block_size
                end_y = start_y + block_size
                rotflag = 0
                if item.orientation[0]==0 and item.orientation[1]==1:
                    rotflag = 0
                elif item.orientation[0]==-1 and item.orientation[1]==0:
                    rotflag = 3
                elif item.orientation[0]==0 and item.orientation[1]==-1:
                    rotflag = 2
                elif item.orientation[0]==1 and item.orientation[1]==0:
                    rotflag = 1
                self.origin_image[start_y:end_y, start_x:end_x, :] = numpy.rot90(item.image, rotflag)
        if self.args.ego_centric:
            side_radius = min(self.args.visible_radius_unit_side, max(width - 1, height - 1))
            front_radius = min(self.args.visible_radius_unit_front, max(width - 1, height - 1))
            radius = max(side_radius, front_radius)
            self.image = numpy.full(((height+2*radius) * block_size,
                                     (width+2*radius) * block_size, 3), 0.5)
            agent_id = self.xmap.item_name_map[agent.name]
            agent_location = self.xmap.items[agent_id].location
            agent_orientation = self.xmap.items[agent_id].orientation
            self.image[radius*block_size:(radius+height)*block_size, radius*block_size:(radius+width)*block_size, :] = self.origin_image

            rotflag = 0
            if agent_orientation[0]==0 and agent_orientation[1]==1:
                rotflag = 0
                start_x = radius + agent_location[0]
                start_y = radius + agent_location[1]
            elif agent_orientation[0]==-1 and agent_orientation[1]==0:
                rotflag = 1
                start_x = radius + agent_location[1]
                start_y = radius + width - agent_location[0]-1
            elif agent_orientation[0]==0 and agent_orientation[1]==-1:
                rotflag = 2
                start_x = radius + width - agent_location[0]-1
                start_y = radius + height - agent_location[1]-1
            elif agent_orientation[0]==1 and agent_orientation[1]==0:
                rotflag = 3
                start_x = radius + height - agent_location[1]-1
                start_y = radius + agent_location[0]

            self.image = numpy.rot90(self.image, rotflag)[(start_y)*block_size:(start_y+front_radius+1)*block_size, (start_x-side_radius)*block_size:(start_x+side_radius+1)*block_size, :]
            for i in range(2*side_radius+1):
                for j in range(1,front_radius+1):
                    if self.inner_state[j,i]==17:
                        self.image[(j+1)*block_size:,i*block_size:(i+1)*block_size,:] = 0.2
                        break
        else:
            self.image = self.origin_image
