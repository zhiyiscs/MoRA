export CUDA_VISIBLE_DEVICES=0
python run.py with data_root="/ssd4tb/datasets/ODIR/" \
        num_gpus=1 \
        num_nodes=1 \
        per_gpu_batchsize=4 \
        task_finetune_ODIR \
        load_path="/ssd4tb/datasets/missing_datasets/pre_trian/vilt_200k_mlm_itm.ckpt"\
        exp_name=ODIR_text_missing07_MSP \
        
        
        
