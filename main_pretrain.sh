python3 main_pretrain.py \
    --dataset cifar10 \
    --train_data_path /data/tmp/cifar/dataset/TRAIN_DIR \
    --val_data_path /data/tmp/cifar/dataset/VAL_DIR \
    --backbone resnet18 \
    --max_epochs 1 \
    --devices 1\
    --accelerator "gpu"\
    --num_workers 26 \
    --precision bf16 \
    --optimizer sgd \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 8 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 \
    --solarization_prob 0.0 \
    --name byol-cifar10 \
    --project self-superivsed \
    --wandb \
    --save_checkpoint \
    --method byol \
    --proj_hidden_dim 2048 \


    #--scale_loss 0.1
# -------------------------------------------------- Documentation usage 


# usage: main_pretrain.py [-h] --dataset {cifar10,cifar100,stl10,imagenet,imagenet100,custom} --train_data_path TRAIN_DATA_PATH
#                         [--val_data_path VAL_DATA_PATH] [--data_format {image_folder,dali,h5}] [--data_fraction DATA_FRACTION]
#                         [--num_crops_per_aug NUM_CROPS_PER_AUG [NUM_CROPS_PER_AUG ...]] --brightness BRIGHTNESS [BRIGHTNESS ...] --contrast
#                         CONTRAST [CONTRAST ...] --saturation SATURATION [SATURATION ...] --hue HUE [HUE ...]
#                         [--color_jitter_prob COLOR_JITTER_PROB [COLOR_JITTER_PROB ...]] [--gray_scale_prob GRAY_SCALE_PROB [GRAY_SCALE_PROB ...]]
#                         [--horizontal_flip_prob HORIZONTAL_FLIP_PROB [HORIZONTAL_FLIP_PROB ...]]
#                         [--gaussian_prob GAUSSIAN_PROB [GAUSSIAN_PROB ...]] [--solarization_prob SOLARIZATION_PROB [SOLARIZATION_PROB ...]]
#                         [--equalization_prob EQUALIZATION_PROB [EQUALIZATION_PROB ...]] [--crop_size CROP_SIZE [CROP_SIZE ...]]
#                         [--min_scale MIN_SCALE [MIN_SCALE ...]] [--max_scale MAX_SCALE [MAX_SCALE ...]] [--debug_augmentations] [--no_labels]
#                         [--mean MEAN [MEAN ...]] [--std STD [STD ...]] [--logger [LOGGER]] [--checkpoint_callback [CHECKPOINT_CALLBACK]]
#                         [--enable_checkpointing [ENABLE_CHECKPOINTING]] [--default_root_dir DEFAULT_ROOT_DIR]
#                         [--gradient_clip_val GRADIENT_CLIP_VAL] [--gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM]
#                         [--process_position PROCESS_POSITION] [--num_nodes NUM_NODES] [--num_processes NUM_PROCESSES] [--devices DEVICES]
#                         [--gpus GPUS] [--auto_select_gpus [AUTO_SELECT_GPUS]] [--tpu_cores TPU_CORES] [--ipus IPUS]
#                         [--log_gpu_memory LOG_GPU_MEMORY] [--progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE]
#                         [--enable_progress_bar [ENABLE_PROGRESS_BAR]] [--overfit_batches OVERFIT_BATCHES] [--track_grad_norm TRACK_GRAD_NORM]
#                         [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--fast_dev_run [FAST_DEV_RUN]]
#                         [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--max_epochs MAX_EPOCHS] [--min_epochs MIN_EPOCHS]
#                         [--max_steps MAX_STEPS] [--min_steps MIN_STEPS] [--max_time MAX_TIME] [--limit_train_batches LIMIT_TRAIN_BATCHES]
#                         [--limit_val_batches LIMIT_VAL_BATCHES] [--limit_test_batches LIMIT_TEST_BATCHES]
#                         [--limit_predict_batches LIMIT_PREDICT_BATCHES] [--val_check_interval VAL_CHECK_INTERVAL]
#                         [--flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS] [--log_every_n_steps LOG_EVERY_N_STEPS] [--accelerator ACCELERATOR]
#                         [--strategy STRATEGY] [--sync_batchnorm [SYNC_BATCHNORM]] [--precision PRECISION]
#                         [--enable_model_summary [ENABLE_MODEL_SUMMARY]] [--weights_summary WEIGHTS_SUMMARY]
#                         [--weights_save_path WEIGHTS_SAVE_PATH] [--num_sanity_val_steps NUM_SANITY_VAL_STEPS]
#                         [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--profiler PROFILER] [--benchmark [BENCHMARK]]
#                         [--deterministic [DETERMINISTIC]] [--reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS]
#                         [--auto_lr_find [AUTO_LR_FIND]] [--replace_sampler_ddp [REPLACE_SAMPLER_DDP]] [--detect_anomaly [DETECT_ANOMALY]]
#                         [--auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]] [--prepare_data_per_node [PREPARE_DATA_PER_NODE]] [--plugins PLUGINS]
#                         [--amp_backend AMP_BACKEND] [--amp_level AMP_LEVEL] [--move_metrics_to_cpu [MOVE_METRICS_TO_CPU]]
#                         [--multiple_trainloader_mode MULTIPLE_TRAINLOADER_MODE] [--stochastic_weight_avg [STOCHASTIC_WEIGHT_AVG]]
#                         [--terminate_on_nan [TERMINATE_ON_NAN]] [--method METHOD]