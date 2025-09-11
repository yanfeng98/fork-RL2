torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    train_data.prompts_per_rollout=64 \
    actor.model_name=Qwen/Qwen3-1.7B-Base \
    actor.max_length_per_device=8192 \
    rollout.train_sampling_params.max_new_tokens=4096 \
    rollout.env_path=envs/gem_env.py \
    rollout.max_turns=16 \
    adv.global_norm=true \
    adv.norm_var=true \
    trainer.project=GEM \
    trainer.experiment_name=language-games_qwen3-1.7b_reinforce \
    trainer.n_epochs=512 \
    trainer.save_freq=64