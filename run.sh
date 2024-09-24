datapath_image="Dataset/Images/"
datapath_mask="Dataset/Masks/"
check=""
model_name="UNETR"
python3 main.py \
    --model $model_name \
    --pretrained \
    --model_checkpoint_test "Results/Normal/Results_UNETR_256/Models/UNETR_256_0_best.pth" \
    --model_checkpoint_test "Results/Normal/Results_UNETR_256/Models/UNETR_256_1_best.pth" \
    --model_checkpoint_test "Results/Normal/Results_UNETR_256/Models/UNETR_256_2_best.pth" \
    --only_forward \
    --attention_weights \
    --data_dir1 $datapath_image \
    --data_dir2 $datapath_mask \
    --batch_size  1 \
    --patch_size "256,256,64" \
    --patch_vit "4,4,2" \
    --overlap 50 \
    --epochs 100 \
    --learning_rate 0.0001 \
    --patience 13


