class config():
    # env config
    render_train     = False
    render_test      = False
    env_name         = "xworld"
    overwrite_render = True
    record           = True
    high             = 1.

    # output config
    output_path  = "results/reactive_agent_0_8_8/"
    model_output = output_path + "model.weights/model"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"

    # model and training config
    num_episodes_test = 500 # 100 how many episodes to use while computing test rewards
    grad_clip         = True
    clip_val          = 10 # 10
    saving_freq       = 20000 # steps 250000
    log_freq          = 100 # steps, should be multiple of learning_freq
    eval_freq         = 20000 # steps between two evaluation 250000
    soft_epsilon      = 0.0

    # nature paper hyper params
    nsteps_train       = 2000000 # 5000000 # 200000
    batch_size         = 16
    buffer_size        = 200000 # 1000000
    target_update_freq = 500 # steps
    gamma              = 0.99
    learning_freq      = 4 # how many steps between two updates
    state_history      = 16
    skip_frame         = 4
    lr_begin           = 0.001
    lr_end             = 0.0001 # 0.0001
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = nsteps_train/2 # 100000
    learning_start     = 1000 # 50000

    # restore config
    restore_param = False
    restore_t = 720000
    deploy_only = False

    visible_radius_unit_side = 2
    visible_radius_unit_front = 4 # 4
    map_config_file = './maps/empty_room_random_object.json'
    ego_centric = True
    h_size = 80 # 200 for embedding
