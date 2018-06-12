class config():
    # env config
    render_train     = False
    render_test      = False
    env_name         = "rcopy"
    overwrite_render = True
    record           = True
    high             = 1.

    # output config
    output_path  = "results/rcopy_0/"
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
    nsteps_train       = 100000 # 5000000 # 200000
    batch_size         = 1
    buffer_size        = 200000 # 1000000
    target_update_freq = 500 # steps
    gamma              = 0.99
    learning_freq      = 1 # how many steps between two updates
    vec_len            = 6
    seq_len            = 24
    skip_frame         = 4
    lr_begin           = 1e-4
    lr_end             = 1e-4 # 0.0001
    lr_nsteps          = nsteps_train/2
    beta_begin          = 0.8
    beta_end            = 0.8
    beta_nsteps         = nsteps_train/2 # 100000
    learning_start     = 0 # 50000

    # restore config
    restore_param = True
    restore_t = 3000
    deploy_only = True

    visible_radius_unit_side = 1
    visible_radius_unit_front = 4
    map_config_file = './maps/four_room_random_apple.json'
    ego_centric = True
    
    # repeat copy related
    num_bits = 4
    min_length = 1
    max_length = 2
    min_repeats = 1
    max_repeats = 2

    #### mem related ####
    memory_size = 16 # 16
    word_size = 16 # 26
    num_reads = 4
    num_writes = 1
    controller_h_size = 64 # 256
    dnc_clip_val = 20 # maximum abs value of controller and dnc outputs
    dnc_h_size = num_bits+1
