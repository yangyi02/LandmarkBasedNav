import copy
import numpy
import json
import random
from . import xworld_item, xworld_utils, dungeon_gen
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class XWorldMap(object):
    def __init__(self, map_config_file, image_block_size):
        # global xworld properties
        self.item_list, self.item_class_id = xworld_utils.parse_item_list()
        self.orientations = xworld_utils.get_orientation()
        self.item_images = xworld_utils.load_item_images(image_block_size)
        # local xworld map properties
        self.dim_max = []
        self.dim_min = []
        self.dim = []
        self.room_meta = []
        self.layout = []
        self.init_items = []
        self.items = []
        self.item_location_map = {}
        self.item_name_map = {}
        self.init_map_config(map_config_file)

    @staticmethod
    def seed(seed=None):
        random.seed(seed)

    def init_map_config(self, map_config_file):
        logging.info("loading %s", map_config_file)
        map_config = json.load(open(map_config_file))
        self.dim_max = map_config['dimensions_max']  # xworld map height and width
        self.dim_min = map_config['dimensions_min']
        self.dim = dict()
        self.dim['width'] = random.randint(self.dim_min['width'], self.dim_max['width'])
        self.dim['height'] = random.randint(self.dim_min['height'], self.dim_max['height'])
        self.room_meta = map_config['room_meta']
        for item_type, item_conf in map_config['items'].items():
            for i in range(item_conf['number']):
                item = xworld_item.XWorldItem(item_type, i)
                if 'instances' in item_conf:
                    if i < len(item_conf['instances']):
                        # item_name = item_conf['instances'].keys()[i]
                        item_name = list(item_conf['instances'].keys())[i]
                        item_class_name = item_name.split('_')[0]
                        assert (item_class_name in self.item_list[item_type].keys())
                        item_class_id = self.item_class_id[item_class_name]
                        item_location = numpy.array(item_conf['instances'][item_name])
                        item = xworld_item.XWorldItem(item_type, i, item_name, item_class_name, item_class_id, item_location)
                self.init_items.append(item)
        logging.info("finish loading %s", map_config_file)

    def reset(self):
        """
        Reset xworld map
        """
        self.items = copy.deepcopy(self.init_items)
        # set dim
        self.dim = dict()
        self.dim['width'] = random.randint(self.dim_min['width'], self.dim_max['width'])
        self.dim['height'] = random.randint(self.dim_min['height'], self.dim_max['height'])
        # Generate a random layout
        self.reset_layout()
        # Replace items with random names to randomly sampled instances
        self.reset_item()
        # Initialize item corresponding image name
        self.reset_image()
        # Initialize location, make sure it's a valid map
        self.reset_location()
        # Initialize orientation
        self.reset_orientation()
        # Build a mapping from location to item id
        self.build_item_map()

    def copy_map(self, src_map):
        self.dim_max = copy.deepcopy(src_map.dim_max)
        self.dim_min = copy.deepcopy(src_map.dim_min)
        self.dim = copy.deepcopy(src_map.dim)
        self.room_meta = copy.deepcopy(src_map.room_meta)
        self.layout = copy.deepcopy(src_map.layout)
        self.init_items = copy.deepcopy(src_map.init_items)
        self.items = copy.deepcopy(src_map.items)
        self.item_location_map = copy.deepcopy(src_map.item_location_map)
        self.item_name_map = copy.deepcopy(src_map.item_name_map)

    def reset_layout(self):
        self.layout = dungeon_gen.Generator(width=self.dim['width'], height=self.dim['height'], max_rooms=self.room_meta['max_rooms'], min_room_xy=self.room_meta['min_room_xy'],
                 max_room_xy=self.room_meta['max_room_xy'], rooms_overlap=False, random_connections=1, random_spurs=3, tiles=dungeon_gen.CHARACTER_TILES)
        self.layout.gen_level()
        self.layout.gen_tiles_level()
        idx = 0
        for i, row in enumerate(self.layout.tiles_level):
            for j, char in enumerate(row):
                if char==' ' or char=='#':
                    loc = numpy.array([j, i])
                    item = xworld_item.XWorldItem(item_type='block', index=idx, location=loc)
                    self.items.append(item)
                    idx += 1
                elif char=='*':
                    loc = numpy.array([j, i])
                    item = xworld_item.XWorldItem(item_type='goal', index=idx, location=loc)
                    self.items.append(item)
                    idx += 1

        # place one goal in each room
        for room in self.layout.room_list:
            ngoal = (numpy.maximum((room[2]-2),(room[3]-2))/2).astype('int32')
            location_exist = {}
            for i in range(ngoal):
                while 1:
                    x = numpy.random.randint(room[0]+1,room[0]+room[2]-1)
                    y = numpy.random.randint(room[1]+1,room[1]+room[3]-1)
                    loc = numpy.array([x, y])
                    if not tuple(loc) in location_exist:
                        location_exist[tuple(loc)] = True
                        break
                item = xworld_item.XWorldItem(item_type='goal', index=idx, location=loc)
                self.items.append(item)
                idx += 1

        self.items.sort(key=lambda v: v.item_type!='goal')

    def reset_item(self):
        """
        Reset xworld map items
        """
        goal_class_name_list = list(self.item_list['goal'].keys())
        goal_flag = numpy.ones(len(goal_class_name_list)).astype('bool')
        for i, item in enumerate(self.items):
            if item.name == '' or item.class_name == '':
                item_type = item.item_type
                if item_type=='goal':
                    rndidx = numpy.random.choice(numpy.where(goal_flag)[0])
                    item_class_name = goal_class_name_list[rndidx]
                    goal_flag[rndidx] = False
                    if goal_flag.sum()==0:
                        goal_flag += True
                else:
                    item_class_name = random.choice(list(self.item_list[item_type].keys()))
                item_name = str(item_class_name) + '_' + str(item.index)
                self.items[i].name = item_name
                self.items[i].class_name = item_class_name
                self.items[i].class_id = self.item_class_id[item_class_name]
        '''
        for i, item in enumerate(self.items):
            if item.name == '' or item.class_name == '':
                item_type = item.item_type
                item_class_name = random.choice(list(self.item_list[item_type].keys()))
                item_name = str(item_class_name) + '_' + str(item.index)
                self.items[i].name = item_name
                self.items[i].class_name = item_class_name
                self.items[i].class_id = self.item_class_id[item_class_name]
        '''

    def reset_image(self):
        """
        Reset xworld map item images
        """
        for i, item in enumerate(self.items):
            item_type = item.item_type
            item_class_name = item.class_name
            image_id = random.randint(1, int(self.item_list[item_type][item_class_name]))
            image_name = str(item_type) + '_' + str(item_class_name) + '_' + str(image_id) + '.jpg'
            self.items[i].image_name = image_name
            self.items[i].image = self.item_images[image_name]

    def reset_location(self):
        """
        Reset xworld map item locations
        """
        width, height = self.dim['width'], self.dim['height']
        # Collect current specified locations
        location_exist = {}
        for i, item in enumerate(self.items):
            if len(item.location) > 0:
                location = item.location
                # Check locations to make sure they are inside map boundary
                assert not xworld_utils.is_out_border(location, width, height), 'location: ' + str(location) + ' outside of map'
                # Check locations to make sure they are not overlapping
                assert tuple(location) not in location_exist, 'multiple items locate on the same location'
                location_exist[tuple(location)] = True
        # Assign random locations to unspecified items
        # TODO: Make the location sampling more efficient
        # TODO: make the map always a valid map: existing at least on route to goal
        for i, item in enumerate(self.items):
            if len(item.location) == 0:
                location = numpy.array([random.randint(0, width-1), random.randint(0, height-1)])
                while tuple(location) in location_exist or (tuple(location+[1,0]) in location_exist and tuple(location+[-1,0]) in location_exist) or (tuple(location+[0,1]) in location_exist and tuple(location+[0,-1]) in location_exist) or (tuple(location+[1,1]) in location_exist and tuple(location+[-1,-1]) in location_exist) or (tuple(location+[1,-1]) in location_exist and tuple(location+[-1,1]) in location_exist):
                    location = numpy.array([random.randint(0, width-1), random.randint(0, height-1)])
                self.items[i].location = location
                location_exist[tuple(location)] = True

    def reset_orientation(self):
        """
        Reset xworld map item orientations
        """
        width, height = self.dim['width'], self.dim['height']
        for i, item in enumerate(self.items):
            if len(item.orientation)==0:
                self.items[i].orientation = self.orientations[0]

    def build_item_map(self):
        self.item_location_map = {}
        for i, item in enumerate(self.items):
            if not item.is_removed:
                if not tuple(item.location) in self.item_location_map:
                    self.item_location_map[tuple(item.location)] = [i]
                else:
                    self.item_location_map[tuple(item.location)].append(i)
        self.item_name_map = {}
        for i, item in enumerate(self.items):
            assert item.name not in self.item_name_map, "error, duplicate item names for multiple objects"
            if not item.is_removed:
                self.item_name_map[item.name] = i

    def step(self, agent, action):
        agent_id = self.item_name_map[agent.name]
        self.items[agent_id].update_orientation(action)
        velocity = agent.velocity[action]
        self.update_location(agent_id, velocity)
        self.build_item_map()
        self.clean()
        self.build_item_map()

    def update_location(self, item_id, velocity):
        next_location = self.items[item_id].get_next_location(velocity)
        if xworld_utils.is_out_border(next_location, self.dim['width'], self.dim['height']):
            return False
        if tuple(next_location) in self.item_location_map:
            for next_item_id in self.item_location_map[tuple(next_location)]:
                next_item_type = self.items[next_item_id].item_type
                if not self.items[item_id].is_movable(next_item_type):
                    return False
                if self.items[next_item_id].is_movable():
                    if not self.update_location(next_item_id, velocity):
                        return False
        self.items[item_id].location = next_location
        return True

    def clean(self):
        """
        Clean items if they are on the same location
        Rebuild item_map after cleaning
        """
        for x in range(self.dim['width']):
            for y in range(self.dim['height']):
                location = numpy.array([x, y])
                if tuple(location) in self.item_location_map:
                    self.clean_items(location)

    def clean_items(self, location):
        """
        Clean items if they are on the same location
        """
        item_ids = self.item_location_map[tuple(location)]
        if len(item_ids) == 1:
            return False
        for item_id in item_ids:
            for next_item_id in item_ids:
                if item_id == next_item_id:
                    continue
                next_item_type = self.items[next_item_id].item_type
                if self.items[item_id].to_be_removed(next_item_type):
                    self.items[item_id].is_removed = True
                    break
        return True
