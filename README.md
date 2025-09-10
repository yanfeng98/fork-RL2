# RL2: Ray Less Reinforcement Learning

A concise library of post-training for large language models.

Despite the simplicity, you should be able to scale up to moderate-sized, *e.g.*, 72B, language models with

* Training engine partition via [Fully Sharded Data Parallelism](https://docs.pytorch.org/docs/stable/fsdp.html) and [Tensor Parallelism](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
* Sequence partition via [Llama Context Parallelism](https://github.com/zhuzilin/ring-flash-attention)
* Inference engine and KV cache partition via Tensor Parallelism

We also support

* Balanced sequence packing for higher throughput
* Multi-turn rollout with [SGLang](https://github.com/sgl-project/sglang) async inference engine
* [GEM](https://github.com/axon-rl/gem.git) (OpenAI Gym like) Agentic Environments

RL2 is a production-ready library! Check our wandb report on [SkyworkRM](https://wandb.ai/chenmientan/SkyworkRM_archive), [UltraFeedback](https://wandb.ai/chenmientan/UltraFeedback_archive), [TinyZero](https://wandb.ai/chenmientan/Countdown_archive).

## Getting Started

### Installation

```bash
git clone git@github.com:yanfeng98/fork-RL2.git
cd fork-RL2
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch==2.6.0 setuptools
uv pip install -e . --no-build-isolation
```

### Data Preperation [[Examples]](https://huggingface.co/Chenmien/datasets)

Hugging Face dataset and various file types, *i.e.*, JSON, JSONL, CSV, Parquet, and Arrow, are accepted.
All trainers support formats of both raw text and messages.

#### SFT

```json
[
    {
        "prompt": "The capital of China is",
        "response": "Beijing."
    }
]
```
```json
[
    {
        "messages": [
            {"role": "user", "content": "What is the capital of China?"},
            {"role": "assistant", "content": "Beijing."}
        ]
    }
]
```
Multi-turn is only supported by the latter format.
#### RM and DPO
```json
[
    {
        "prompt": "The capital of China is",
        "chosen": "Beijing.",
        "rejected": "Shanghai."
    }
]
```

```json
[
    {
        "messages": [
            {"role": "user", "content": "What is the capital of China?"}
        ],
        "chosen": "Beijing.",
        "rejected": "Shanghai."
    }
]
```
#### PPO
```json
[
    {
        "prompt": "The capital of China is",
        "extra_info": {
            "answer": "Beijing"
        }
    }
]
```
```json
[
    {
        "messages": [
            {"role": "user", "content": "What is the capital of China?"}
        ],
        "extra_info": {
            "answer": "Beijing"
        }
    }
]
```

### Environments [[Examples]](./envs)

In PPO, the language model interacts with the environment through a user-defined function `step` in the following format.
```python
async def step(
    state: str, action: str, extra_info: Dict
) -> Dict:
    action_type = parse_action_type(action)
    env_response = {
        "next_state": None,
        "reward": 0.0,
        "score": 0.0,
        "done": False,
        "extra_info": extra_info
    }
    if action_type == "search":
        query = parse_query(action)
        passage = await search_result(query)
        env_response["next_state"] = state + action + passage
    elif action_type == "answer":
        pred = parse_pred(action)
        reward = float(is_equivalent(pred, extra_info["answer"]))
        env["reward"] = reward
        env["score"] = score
        env_response["done"] = True
    return env_response
```
* `state` and `action` are the input and output of language model in the last turn and `next_state` is the input of language model in the next turn.
When `state + action` is a prefix of `next_state`, the two turns will be processed in a single sequence.
* `reward` is used to compute advantages (and subsequently update the model) while `score` is used to log the model performance.
Diverge values may be used when needed.
* `done` indicates whether to proceed to the next turn.
* `extra_info` contains everything not aforementioned, *e.g.*, answer.

The function should be included in a Python script where the path is specified by `actor.rollout.env_path`.

### Launch [[Examples]](./examples)

Use `torchrun` to launch the trainer. For example, for single node
```bash
torchrun \
    --nproc_per_node=<number of GPUs> \
    -m RL2.trainer.ppo \
    <args>
```
For multi nodes
```bash
torchrun \
    --nnodes=<number of nodes> \
    --node_rank=<rank of node> \
    --nproc_per_node=<number of GPUs on a node> \
    --master_addr=<address of master node> \
    --master_port=<port of master node> \
    -m RL2.trainer.ppo \
    <args>
```

## Hyper-Parameters

### Training Engine Partition

By default, *i.e.*, `ddp_size=1, tp_size=1`, your model will be partitioned via ZeRO stage 3.
`ddp_size` specifies the number of model parameter copies.
Larger `ddp_size` leads to higher memory consumption and lower communication cost.
For large models, you may specify `tp_size > 1` to enable tensor parallelism.
The product of `ddp_size` and `tp_size` should be a factor of the total number of GPUs.

### Sequence Length

For SFT, RM, and DPO, `max_length` is used to truncate sequences.
In RM and DPO, the chosen and rejected sequences will be packed together, so the actual sequence length can be up to twice of `max_length`.
For PPO, `max_new_tokens` is used to terminate generations.
The length of any sequence cannot exceed `sp_size * tp_size * max_length_per_device`.

### Algorithm

The default algorithm is [Dr. GRPO](https://arxiv.org/abs/2503.20783), where the loss is averaged at the token level and the advantage is not divided by the standard deviation.

* To use OpenAI PPO, set `kl.type=reward`, `kl.reward_estimator=k1`, and `adv.estimator=gae`
* To use DeepSeek GRPO, set `actor.avg_level=sequence`, `kl.type=loss`, `kl.loss_estimator=k3`, and `adv.norm_var=true`
