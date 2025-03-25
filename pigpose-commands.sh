python demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/rtmdet_m_8xb32-300e_coco.py https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth --input tests/data/pigpose/pig.png --show --draw-heatmap --det-cat-id=15

python demo/inferencer_demo.py tests/data/ap10k --pose2d animal --vis-out-dir vis_results/ap10k

# For showing annotatioons
# https://mmpose.readthedocs.io/en/latest/user_guides/prepare_datasets.html
python tools/misc/browse_dataset.py configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py --mode original
python tools/misc/browse_dataset.py configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py --mode transformed

# NOTE: If you get an error, you need to rebuild the packages
# https://stackoverflow.com/questions/70136275/problem-encountered-in-mmdetection-keyerror-mask-detectiondataset-is-not-in-t
# python setup.py install
# pip install -v -e .
python tools/misc/browse_dataset.py configs/animal_2d_keypoint/topdown_heatmap/pigpose/td-hm_hrnet-w32_8xb64-210e_pigpose-256x256.py --mode transformed

# Training
# NOTE: if training in  laptop, you may need to reduce the batch size (32) 
python tools/train.py configs/animal_2d_keypoint/topdown_heatmap/pigpose/td-hm_hrnet-w32_8xb64-210e_pigpose-256x256.py

# Training with multiple GPUs
PORT=29501 bash ./tools/dist_train.sh configs/animal_2d_keypoint/topdown_heatmap/pigpose/td-hm_hrnet-w32_8xb64-210e_pigpose-256x256.py 2 

# Test visualization
python tools/test.py configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth --show

python tools/test.py configs/animal_2d_keypoint/topdown_heatmap/pigpose/td-hm_hrnet-w32_8xb64-210e_pigpose-256x256.py work_dirs/td-hm_hrnet-w32_8xb64-210e_pigpose-256x256/best_coco_AP_epoch_180.pth --show

# Get results
python tools/test.py configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth --work-dir work_dirs/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256/eval

python tools/test.py configs/animal_2d_keypoint/topdown_heatmap/pigpose/td-hm_hrnet-w32_8xb64-210e_pigpose-256x256.py work_dirs/td-hm_hrnet-w32_8xb64-210e_pigpose-256x256/best_coco_AP_epoch_3.pth --work-dir work_dirs/td-hm_hrnet-w32_8xb64-210e_pigpose-256x256/eval

python demo/bottomup_demo.py configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512.py https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512_ac7c17bf-20221228.pth --input tests/data/coco/000000197388.jpg --output-root=vis_results --show --save-predictions

python demo/bottomup_demo.py configs/animal_2d_keypoint/cid/pigpose/cid_hrnet-w32_8xb20-140e_pigpose-512x512.py work_dirs/cid_hrnet-w32_8xb20-140e_coco-512x512/best_coco_AP_epoch_130.pth --input tests/data/coco/000000197388.jpg --output-root=vis_results --show --save-predictions


python demo/bottomup_demo.py configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512.py https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512_ac7c17bf-20221228.pth --input tests/data/coco/000000197388.jpg --output-root=vis_results --show --save-predictions

# python demo/bottomup_demo.py configs/animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_ap10k_256x256-18aac840_20211029.pth --input "tests/data/pigpose/melvin-samples/Nursery_1 (145).png" --output-root=vis_results --show --save-predictions

# Get predictions
python demo/inferencer_demo.py tests/data/pigpose/melvin-samples --pose2d animal --pred-out-dir vis_results/melvin-results --vis-out-dir vis_results/melvin-results


