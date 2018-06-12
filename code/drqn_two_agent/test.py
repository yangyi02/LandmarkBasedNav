import numpy
from collections import deque

def compute_shortest_path():
        width, height = 8, 8
        # only empty space is walkable
        visit_map = numpy.random.random([height, width])>0.8
        print(visit_map)
        while 1:
            goal_location = numpy.array([numpy.random.randint(width),numpy.random.randint(height)])
            if not visit_map[goal_location[1], goal_location[0]]:
                break
        # action map -1block 0-> 1^ 2<- 3_
        action_map = -numpy.ones([height, width])
        dist_map = numpy.inf*numpy.ones([height, width])
        dist_map[goal_location[1], goal_location[0]] = 0
        d = deque()
        d.append(goal_location)
        while d:
            cur_loc = d.popleft()
            visit_map[cur_loc[1], cur_loc[0]] = True
            offsets = [numpy.array([0, -1]), numpy.array([1, 0]), numpy.array([0, 1]), numpy.array([-1, 0])]
            actions = [3, 2, 1, 0]
            for i in range(4):
                new_loc = cur_loc+offsets[i]
                if (new_loc>=0).sum()==2 and (new_loc<[width, height]).sum()==2:
                    if not visit_map[new_loc[1], new_loc[0]]:
                        cur_dist = dist_map[cur_loc[1], cur_loc[0]]+1+(action_map[cur_loc[1], cur_loc[0]]!=actions[i])
                        if cur_dist<dist_map[new_loc[1], new_loc[0]]:
                            dist_map[new_loc[1], new_loc[0]] = cur_dist
                            action_map[new_loc[1], new_loc[0]] = actions[i]
                        d.append(new_loc)
        print(goal_location)
        print(dist_map)
        print(action_map)
        print(visit_map)

if __name__ == '__main__':
        compute_shortest_path()