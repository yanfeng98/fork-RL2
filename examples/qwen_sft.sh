torchrun \
    --nproc_per_node=1 \
    -m RL2.trainer.sft \
    data.path=Chenmien/LIMO \
    data.max_length=1024 \
    data.batch_size=4 \
    actor.model_name=Qwen/Qwen2.5-0.5B-Instruct \
    actor.sp_size=1 \
    actor.max_length_per_device=1024 \
    trainer.use_wandb=false \
    trainer.project=LIMO \
    trainer.experiment_name=qwen2.5-0.5b-inst \
    trainer.n_epochs=1