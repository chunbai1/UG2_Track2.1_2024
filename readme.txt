First use the TMT dataset to fine-tune the DATUM model 40k iter. Please put the data at data/TMT/
```
CUDA_VISIBLE_DEVICES=0,1 python train_DATUM_static.py --train_path ../../data/TMT/train_turb/ --val_path ../../data/TMT/test_turb/ -f ../DATUM_dynamic.pth --log_path ../experiments/turb_static_dataset/ --start_over --iters 40000 --batch-size 2 --patch-size 512 --learning-rate 0.0001 --num_frames 50 --num_gpus 2 --other_mod blur
```

Then, the trained weights are used for two inferences. Use 90 frames patch-size 512 at a time, use 100 frames patch-size 360 at a time.
```
python test_DATUM_static.py --loaded_frames 90 --valid_frames 1 --load  ../experiments/turb_static_dataset/checkpoints/model_40000.pth --val_path ../../data/final_data/ --result_path ../../data/result/experiment_turb_static_dataset_iter40k_fs5to95_out1/ --start_frame 5 --total_frames 100
python test_DATUM_static.py --loaded_frames 100 --valid_frames 1 --load  ../experiments/turb_static_dataset/checkpoints/model_40000.pth --val_path ../../data/final_data/ --result_path ../../data/result/experiment_turb_static_dataset_iter40k_ps360_fs100_out1/ --start_frame 0 --total_frames 100 --patch-size 360
```

Then use the multi_model_fusion.py script to perform a 7:3 weighted average of 100 frames at 360ps and 90 frames at 512ps.
```
python multi_model_fusion.py --img_folders data/result/experiment_turb_static_dataset_iter40k_ps360_fs100_out1/ data/result/experiment_turb_static_dataset_iter40k_fs5to95_out1/ --weights 0.7 0.3 --save_path data/result/model_fusion_73/
```

Finally, use MPRNet's deblurring to deblur the weighted average result.
```python demo.py --input_dir ../data/resul/model_fusion_73/ --result_dir ../data/result/model_fusion_73_MPRNet_deblur/ --task Deblurring
```
