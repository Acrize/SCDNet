export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u -m torch.distributed.launch \
    --nproc_per_node=8 --node_rank=0 --master_port=16229 ./train.py \
    --datapath "../datasets" \
    --benchmark coco \
    --fold 0 \
    --bsz 7 \
    --nworker 0 \
    --backbone resnet50 \
    --feature_extractor_path "../backbones/resnet50.pth" \
    --logpath "./logs" \
    --lr 1e-3 \
    --nepoch 200 \
    --test_num 300