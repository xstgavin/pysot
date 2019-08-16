CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=2331 \
    ../../tools/train.py --cfg config.yaml
