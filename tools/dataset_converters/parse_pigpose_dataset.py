import os
import json
import re
import numpy as np
import time

def list_all_files(root_dir, ext='.json'):
    """List all files in the root directory and all its sub directories.

    :param root_dir: root directory
    :param ext: filename extension
    :return: list of files
    """
    files = []
    file_list = os.listdir(root_dir)
    for i in range(0, len(file_list)):
        path = os.path.join(root_dir, file_list[i])
        if os.path.isdir(path):
            files.extend(list_all_files(path))
        if os.path.isfile(path):
            if path.lower().endswith(ext):
                files.append(path)
    return files



def get_anno_info():
    keypoints_info = {
        0: 'Nose',
        1: 'Left_Eye',
        2: 'Right_Eye',
        3: 'Left_Ear',
        4: 'Right_Ear',
        5: 'Left_Shoulder',
        6: 'Right_Shoulder',
        7: 'Left_Elbow',
        8: 'Right_Elbow',
        9: 'Left_Paw',
        10: 'Right_Paw',
        11: 'Left_Hip',
        12: 'Right_Hip',
        13: 'Left_Knee',
        14: 'Right_Knee',
        15: 'Left_Foot',
        16: 'Right_Foot',
        17: 'Tail',
        18: 'Center'
    }

    skeleton_info = {
        0: [0, 1], # 'Nose' - 'Left_Eye'
        1: [0, 2], # 'Nose' - 'Right_Eye'
        2: [1, 2], # 'Left_Eye' - 'Right_Eye'
        3: [1, 3], # 'Left_Eye' - 'Left_Ear'
        4: [2, 4], # 'Right_Eye' - 'Right_Ear'
        5: [0, 18], # 'Nose' - 'Center'
        6: [5, 18], # 'Left_Shoulder' - 'Center'
        7: [6, 18], # 'Right_Shoulder' - 'Center'
        8: [5, 7], # 'Left_Shoulder' - 'Left_Elbow'
        9: [6, 8], # 'Right_Shoulder' - 'Right_Elbow'
        10: [7, 9], # 'Left_Elbow' - 'Left_Paw'
        11: [8, 10], # 'Right_Elbow' - 'Right_Paw'
        12: [11, 13], # 'Left_Hip' - 'Left_Knee'
        13: [12, 14], # 'Right_Hip' - 'Right_Knee'
        14: [13, 15], # 'Left_Knee' - 'Left_Foot'
        15: [14, 16], # 'Right_Knee' - 'Right_Foot'
        16: [11, 17], # 'Left_Hip' - 'Tail'
        17: [12, 17], # 'Right_Hip' - 'Tail'
        18: [17, 18] # 'Tail' - 'Center'
    }
    category_info = [{
        'supercategory': 'pig',
        'id': 1,
        'name': 'pig',
        'keypoints': list(keypoints_info.values()),
        'skeleton': list(skeleton_info.values())
    }]

    return keypoints_info.values(), skeleton_info.values(), category_info

def create_coco_dataset(file_list, img_root, save_path, start_ann_id=0):
    """Save annotations in coco-format.

    :param file_list: list of data annotation files.
    :param img_root: the root dir to load images.
    :param save_path: the path to save transformed annotation file.
    :param start_ann_id: the starting point to count the annotation id.
    :param val_num: the number of annotated objects for validation.
    """

    keypoints_info, _, _ = get_anno_info()

    images = []
    annotations = []
    img_ids = []
    ann_ids = []

    ann_id = start_ann_id

    cat2id = {'pig': 1}
    for file in file_list:
        data_anno = json.load(open(file, 'r'))

        img_id = int(re.findall(r'\d+', data_anno['imagePath'])[0])

        image = {}
        image['id'] = img_id
        image['file_name'] = os.path.join(img_root, data_anno["imagePath"])
        image['height'] = data_anno["imageHeight"]
        image['width'] = data_anno["imageWidth"]

        images.append(image)
        img_ids.append(img_id)

        # prepare annotations
        shapes = data_anno['shapes']
        pig_annotations = {}

        # Get bbox annotations
        bbox_annos = [shape for shape in shapes if shape['label'] == 'bbox']
        for bbox_anno in bbox_annos:
            # General information
            group_id = bbox_anno['group_id']
            anno = {}
            anno['image_id'] = img_id
            anno['id'] = ann_id
            anno['iscrowd'] = 0
            anno['category_id'] = cat2id['pig']

            # Bounding box
            pig_annotations[group_id] = anno
            flattened_bboxpoints = np.array(bbox_anno["points"]).flatten().tolist()
            # Get width and height
            flattened_bboxpoints[2] = flattened_bboxpoints[2] - flattened_bboxpoints[0]
            flattened_bboxpoints[3] = flattened_bboxpoints[3] - flattened_bboxpoints[1]
            pig_annotations[group_id]["bbox"] = flattened_bboxpoints

            ann_id += 1

        # Get segmentation annotations
        segmentation_annos = [shape for shape in shapes if shape['label'] == 'pig']
        for seg_anno in segmentation_annos:
            pig_annotations[group_id]["segmentation"] = seg_anno["points"]
            
        # Initialize keypoints
        for pig_anno in pig_annotations.values():
            pig_anno["keypoints"] = np.zeros([len(keypoints_info), 3], dtype=np.float32)

        # Get keypoint annotations
        keypoint_annos = [shape for shape in shapes if not shape['label'] in ['pig','bbox']]

        for kpt_anno in keypoint_annos:
            keypoint_name = kpt_anno['label']
            keypoint_id = int(keypoint_name)
            # Correct the mistake in original annotations
            if keypoint_id == 18:
                keypoint_id = 17
            elif keypoint_id == 20:
                keypoint_id = 18
                
            keypoint_group_id = kpt_anno['group_id']

            pig_annotations[keypoint_group_id]["keypoints"][keypoint_id, 0] = float(kpt_anno['points'][0][0])
            pig_annotations[keypoint_group_id]["keypoints"][keypoint_id, 1] = float(kpt_anno['points'][0][1])
            pig_annotations[keypoint_group_id]["keypoints"][keypoint_id, 2] = 2

        # Postprocess keypoints
        for pig_anno in pig_annotations.values():
            keypoints = pig_anno['keypoints']
            pig_anno['keypoints'] = keypoints.reshape(-1).tolist()
            pig_anno['num_keypoints'] = int(sum(keypoints[:, 2] > 0))

            annotations.append(pig_anno)
            ann_ids.append(pig_anno['id'])

    # Structure of coco file
    cocotype = {}

    cocotype['info'] = {}
    cocotype['info'][
        'description'] = 'PigPose dataset Generated by KoLab Team'
    cocotype['info']['version'] = '1.0'
    cocotype['info']['year'] = time.strftime('%Y', time.localtime())
    cocotype['info']['date_created'] = time.strftime('%Y/%m/%d',
                                                     time.localtime())
    
    cocotype['images'] = images
    cocotype['annotations'] = annotations

    # cocotype['categories'] = category_info

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json.dump(cocotype, open(save_path, 'w'), indent=4)
    print('=========================================================')
    print('number of images:', len(img_ids))
    print('number of annotations:', len(ann_ids))
    print(f'done {save_path}')




dataset_dir = 'data/pigpose/'

# We choose the images from BemaPig2D-pure (BemaPig2D sepearated into individual files for LabelMe)
# Train: 3008 images
train_dir = os.path.join(dataset_dir, 'train')
create_coco_dataset(list_all_files(train_dir), "train", os.path.join(dataset_dir, 'pigpose_train.json'))

# Val: 332 images
val_dir = os.path.join(dataset_dir, 'eval')
create_coco_dataset(list_all_files(val_dir), "eval", os.path.join(dataset_dir, 'pigpose_val.json'))
