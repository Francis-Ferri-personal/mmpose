python demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/rtmdet_m_8xb32-300e_coco.py https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth --input tests/data/pigpose/pig.png --show --draw-heatmap --det-cat-id=15

python demo/inferencer_demo.py tests/data/ap10k --pose2d animal --vis-out-dir vis_results/ap10k

# For showing annotatioons
# https://mmpose.readthedocs.io/en/latest/user_guides/prepare_datasets.html
python tools/misc/browse_dataset.py .\configs\animal_2d_keypoint\topdown_heatmap\animalpose\td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py --mode original
python tools/misc/browse_dataset.py .\configs\animal_2d_keypoint\topdown_heatmap\animalpose\td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py --mode transformed

# NOTE: If you get an error, you need to rebuild the packages
# https://stackoverflow.com/questions/70136275/problem-encountered-in-mmdetection-keyerror-mask-detectiondataset-is-not-in-t
# python setup.py install
# pip install -v -e .
python tools/misc/browse_dataset.py .\configs\animal_2d_keypoint\topdown_heatmap\pigpose\td-hm_hrnet-w32_8xb64-210e_pigpose-256x256.py --mode transformed

 python tools/train.py .\configs\animal_2d_keypoint\topdown_heatmap\pigpose\td-hm_hrnet-w32_8xb64-210e_pigpose-256x256.py 