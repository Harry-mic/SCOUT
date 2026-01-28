set -e

USE_GRPO="algorithm.adv_estimator=grpo"
USE_PPO="algorithm.adv_estimator=gae" 
USE_BASE="algorithm.kl_ctrl.kl_coef=0.0 actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.rollout.rollout_filter_ratio=0.25"

python train.py --config-name _3_frozen_lake \
    system.CUDA_VISIBLE_DEVICES="'0,1,2,3,4,5,6,7'" \
    trainer.project_name=ragen_latest_qwen2.5_3B_it_explict \
    trainer.n_gpus_per_node=8 \
    model_path= \
    trainer.save_freq=200 \
    trainer.experiment_name=frozenlake_slippery $USE_PPO $USE_BASE 