import argparse
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def parser():
    arg_parser = argparse.ArgumentParser(description='XWorld simulator', add_help=False)
    arg_parser.add_argument('--teacher_type', default='NAVI_GOAL',
                            help='NAVI_GOAL, NAVI_NEAR, NAVI_BOX, PUSH_BOX, ...')
    arg_parser.add_argument('--single_word', action='store_true',
                            help='whether the command is a single word or not')
    arg_parser.add_argument('--show_frame', action='store_true',
                            help='whether show frames during running')
    arg_parser.add_argument('--map_config', default='map_examples/example1.json',
                            help='map config file')
    arg_parser.add_argument('--keep_command', action='store_true',
                            help='whether teacher keep providing command at every step')
    arg_parser.add_argument('--ego_centric', action='store_true',
                            help='whether the image is ego centric')
    arg_parser.add_argument('--visible_radius_unit_side', type=int, default=1,
                            help='robot visible field radius on its side')
    arg_parser.add_argument('--visible_radius_unit_front', type=int, default=4,
                            help='robot visible field radius in its front')
    arg_parser.add_argument('--image_block_size', type=int, default=64,
                            help='image block size per object')
    arg_parser.add_argument('--pause_screen', action='store_true',
                            help='whether pause screen at every step')
    arg_parser.add_argument('--discount_factor', type=float, default=0.99,
                            help='discount factor in game')
    return arg_parser
