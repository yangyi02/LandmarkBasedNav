class config():
    # env config
    render_train     = False
    render_test      = False
    env_name         = "graph_world_small_sp"
    overwrite_render = True
    record           = True
    high             = 1.

    # output config
    output_path  = "results/graph_world_small_sp_simpdnc_1_sharpread/"
    model_output = output_path + "model.weights/model"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"

    # model and training config
    num_episodes_test = 50 # 100 how many episodes to use while computing test rewards
    grad_clip         = True
    clip_val          = 10 # 10
    saving_freq       = 1000 # steps 250000
    log_freq          = 100 # steps, should be multiple of learning_freq
    eval_freq         = 1000 # steps between two evaluation 250000
    soft_epsilon      = 0.0

    # nature paper hyper params
    nsteps_train       = 1000000 # 5000000 # 200000
    batch_size         = 16
    buffer_size        = 200000 # 1000000
    target_update_freq = 500 # steps
    gamma              = 0.99
    learning_freq      = 1 # how many steps between two updates
    state_history      = 24
    skip_frame         = 4
    lr_begin           = 1e-5
    lr_end             = 1e-5 # 0.0001
    lr_nsteps          = nsteps_train/2
    beta_begin          = 0.8
    beta_end            = 0.8
    beta_nsteps         = nsteps_train/2 # 100000
    learning_start     = 0 # 50000

    # restore config
    restore_param = False
    restore_t = 42000
    deploy_only = False

    visible_radius_unit_side = 1
    visible_radius_unit_front = 4
    map_config_file = './maps/four_room_random_apple.json'
    ego_centric = True
    
    #### mem related ####
    memory_size = 20 #128 # 16
    word_size = 10 #50 # 26
    key_size = 10
    num_reads = 1
    num_writes = 1
    controller_h_size = 32 #256 # 256
    dnc_clip_val = 10 # maximum abs value of controller and dnc outputs
    dnc_h_size = word_size+1

    #### graph world related ####
    n_node = 16
    k_ring = 4
    p_rewiring = 0.5
    path_len_limit = 5
    planning_len = 10
    ndigits = 2
    nway = 5
    num_actions = 5
    max_step_len = 5
    use_transition_only_during_answering = False
