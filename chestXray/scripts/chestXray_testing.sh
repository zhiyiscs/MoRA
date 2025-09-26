export CUDA_VISIBLE_DEVICES=2
python run.py with data_root="/ssd4tb/datasets/chestXray/" \
        num_gpus=1 \
        num_nodes=1 \
        per_gpu_batchsize=4 \
        task_finetune_chestXray \
        load_path="result/chestXray_text_missing07_MAP_new_seed0_from_vilt_200k_mlm_itm/version_0/checkpoints/epoch=45-step=34321.ckpt"\
        exp_name=chestXray\
        test_only=True 

