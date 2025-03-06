# python eval/gen_npz.py
# python eval/evaluator.py \
#     --ref_batch datasets/Structured3D/all_data_test_33180.npz \
#     --sample_batch finalexp_img_log/baseline_fov90_epoch20_33180_image_log/epoch20_all_data.npz \
#     --save_result_path paper_result/baseline_fov90_epoch20_33180_all_data.yaml
# python eval/evaluator.py \
#     --ref_batch datasets/Structured3D/all_data_test_33180.npz \
#     --sample_batch finalexp_img_log/clip_contrastivee_dcn_rotate_sgag_fov90_33180_epoch20_image_log/epoch20_all_data.npz \
#     --save_result_path paper_result/clip_contrastivee_dcn_rotate_sgag_fov90_33180_epoch20_all_data.yaml

# python eval/evaluator.py \
#     --ref_batch datasets/Structured3D/all_data_test_55180.npz \
#     --sample_batch finalexp_img_log/baseline_fov90_epoch20_image_log/epoch20_55180_all_data.npz \
#     --save_result_path paper_result/baseline_fov90_epoch20_55180_all_data.yaml
# python eval/evaluator.py \
#     --ref_batch datasets/Structured3D/all_data_test_55180.npz \
#     --sample_batch finalexp_img_log/clip_contrastivee_dcn_rotate_sgag_fov90_55180_epoch20_image_log/epoch20_all_data.npz \
#     --save_result_path paper_result/clip_contrastivee_dcn_rotate_sgag_fov90_55180_epoch20_all_data.yaml

python eval/gen_npz.py

python eval/evaluator.py \
    --ref_batch datasets/Structured3D/all_data.npz \
    --sample_batch /group/30042/jerryxwli/code/T2I-Adapter/outputs/epoch20_all_data.npz \
    --save_result_path paper_result/t2i-adapter_result.yaml

# python eval/evaluator.py \
#     --ref_batch datasets/Structured3D/all_data.npz \
#     --sample_batch finalexp_img_log/clip_contrastivee_dcn_rotate_sgag_fov30_epoch20_image_log/epoch20_all_data.npz \
#     --save_result_path paper_result/clip_contrastivee_dcn_rotate_sgag_fov30_epoch20_all_data.yaml