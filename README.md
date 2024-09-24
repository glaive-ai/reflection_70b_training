# Run training for Reflection 70B

Install the requirements and you can use this command to launch training. It works in both single node and multi node environments.

```bash
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nnodes 1 --node_rank 0 --master_addr {master ip address} --master_port {master port} --nproc_per_node=8 train.py --config config.yaml
```