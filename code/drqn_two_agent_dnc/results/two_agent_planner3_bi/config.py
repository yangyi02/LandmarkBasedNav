class config():
    # env config
    render_train     = False
    render_test      = False
    env_name         = "xworld"
    overwrite_render = True
    record           = True
    high             = 1.

    # output config
    output_path  = "results/two_agent_planner3_bi/"
    model_output = output_path + "model.weights/model"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"

    # model and training config
    num_episodes_test = 50 # 100 how many episodes to use while computing test rewards
    grad_clip         = True
    clip_val          = 10 # 10
    saving_freq       = 1000 # steps 250000
    log_freq          = 50 # steps, should be multiple of learning_freq
    eval_freq         = 1000 # steps between two evaluation 250000
    soft_epsilon      = 0.0

    # nature paper hyper params
    nsteps_train       = 1000000 # 5000000 # 200000
    batch_size         = 16
    buffer_size        = 5000000 # 1000000
    target_update_freq = 500 # steps
    gamma              = 0.99
    learning_freq      = 1 # how many steps between two updates
    state_history      = 24
    skip_frame         = 4
    lr_begin           = 1e-5
    lr_end             = 1e-5
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = nsteps_train/2 # 100000
    learning_start     = 0 # 50000

    # restore config
    restore_param = False
    restore_t = 2000 # 8x8
    deploy_only = False
    vis_heat_map = False

    visible_radius_unit_side = 2
    visible_radius_unit_front = 4 # 4
    map_config_file = './maps/two_room.json'
    ego_centric = True
    h_size = 80 # 200 for embedding

    #### mem related ####
    memory_size = 12 #128 # 16
    word_size = 96 #50 # 26
    num_reads = 5
    num_writes = 1
    controller_h_size = 128 #256 # 256
    dnc_clip_val = 10 # maximum abs value of controller and dnc outputs
    dnc_h_size = 128

    ndigits = 2
    nway = 5
    nins = 2