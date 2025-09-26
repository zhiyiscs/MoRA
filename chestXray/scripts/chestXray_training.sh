export CUDA_VISIBLE_DEVICES=1
python run.py with data_root="/ssd4tb/datasets/chestXray/" \
        num_gpus=1 \
        num_nodes=1 \
        per_gpu_batchsize=4 \
        task_finetune_chestXray \
        load_path="/ssd4tb/datasets/missing_datasets/pre_trian/vilt_200k_mlm_itm.ckpt"\
        exp_name=chestXray_both_missing07_MAP_new \
        
        
