torchrun \
    --nproc_per_node=1 \
    -m RL2.trainer.ppo \
    train_data.path=train@Chenmien/Countdown \
    train_data.prompts_per_rollout=16 \
    train_data.responses_per_prompt=4 \
    test_data.path=test@Chenmien/Countdown \
    actor.model_name=Qwen/Qwen2.5-0.5B-Instruct \
    actor.max_length_per_device=4096 \
    rollout.gpu_memory_utilization=0.2 \
    rollout.train_sampling_params.max_new_tokens=1024 \
    "rollout.train_sampling_params.stop=['</answer>']" \
    rollout.env_path=envs/countdown.py \
    trainer.use_wandb=false \
    trainer.project=Countdown \
    trainer.experiment_name=qwen2.5-0.5b_reinforce \
    trainer.test_freq=8 \
    trainer.save_freq=32