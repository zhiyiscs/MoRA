export CUDA_VISIBLE_DEVICES=0
python run.py with data_root="/ssd4tb/datasets/ODIR/" \
        num_gpus=1 \
        num_nodes=1 \
        per_gpu_batchsize=4 \
        task_finetune_ODIR \
        load_path="/ssd4tb/Zhiyi/ODIR/result/ODIR_both_missing07_baseline_seed0_from_vilt_200k_mlm_itm/version_0/checkpoints/epoch=18-step=809.ckpt"\
        exp_name=ODIR_both_missing07 \
        test_only=True 

