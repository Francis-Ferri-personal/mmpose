dataset_info = dict(
    dataset_name='pigpose',
    paper_info=dict(
        author='Francis, Melvin, Reza',
        title='',
        container='',
        year='2025',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(
            name='Nose', 
            id=0, color=[0, 255, 0], 
            type='upper', 
            swap=''),
        1:
        dict(
            name='Left_Eye',
            id=1,
            color=[255, 128, 0],
            type='upper',
            swap='Right_Eye'),
        2:
        dict(
            name='Right_Eye',
            id=2,
            color=[0, 255, 0],
            type='upper',
            swap='Left_Eye'),
        3:
        dict(
            name='Left_Ear',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap='Right_Ear'),
        4:
        dict(
            name='Right_Ear', 
            id=4, 
            color=[51, 153, 255], 
            type='upper', 
            swap='Left_Ear'),
        5:
        dict(
            name='Left_Shoulder', 
            id=5, 
            color=[51, 153, 255], 
            type='upper', 
            swap='Right_Shoulder'),
        6:
        dict(
            name='Right_Shoulder', 
            id=6, 
            color=[51, 153, 255], 
            type='upper',
            swap='Left_Shoulder'),
        7:
        dict(
            name='Left_Elbow', 
            id=7, 
            color=[51, 153, 255], 
            type='upper', 
            swap='Right_Elbow'),
        8:
        dict(
            name='Right_Elbow',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap='Left_Elbow'),
        9:
        dict(
            name='Left_Paw',
            id=9,
            color=[255, 128, 0],
            type='upper',
            swap='Right_Paw'),
        10:
        dict(
            name='Right_Paw',
            id=10,
            color=[0, 255, 0],
            type='upper',
            swap='Left_Paw'),
        11:
        dict(
            name='Left_Hip',
            id=11,
            color=[255, 128, 0],
            type='lower',
            swap='Right_Hip'),
        12:
        dict(
            name='Right_Hip',
            id=12,
            color=[0, 255, 0],
            type='lower',
            swap='Left_Hip'),
        13:
        dict(
            name='Left_Knee',
            id=13,
            color=[255, 128, 0],
            type='lower',
            swap='Right_Knee'),
        14:
        dict(
            name='Right_Knee',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap='Left_Knee'),
        15:
        dict(
            name='Left_Foot',
            id=15,
            color=[255, 128, 0],
            type='lower',
            swap='Right_Foot'),
        16:
        dict(
            name='Right_Foot',
            id=16,
            color=[0, 255, 0],
            type='lower',
            swap='Left_Foot'),
        17:
        dict(
            name='Tail',
            id=17,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        18:
        dict(
            name='Center',
            id=18,
            color=[0, 255, 0],
            type='lower',
            swap='')
    },
    skeleton_info={
        0: dict(link=('Nose', 'Left_Eye'), id=0, color=[51, 153, 255]),
        1: dict(link=('Nose', 'Right_Eye'), id=1, color=[0, 255, 0]),
        2: dict(link=('Left_Eye', 'Right_Eye'), id=2, color=[255, 128, 0]),
        3: dict(link=('Left_Eye', 'Left_Ear'), id=3, color=[0, 255, 0]),
        4: dict(link=('Right_Eye', 'Right_Ear'), id=4, color=[255, 128, 0]),
        5: dict(link=('Nose', 'Center'), id=5, color=[51, 153, 255]),
        6: dict(link=('Left_Shoulder', 'Center'), id=6, color=[51, 153, 255]),
        7: dict(link=('Right_Shoulder', 'Center'), id=7, color=[51, 153, 255]),
        8: dict(link=('Left_Shoulder', 'Left_Elbow'), id=8, color=[0, 255, 0]),
        9: dict(link=('Right_Shoulder', 'Right_Elbow'), id=9, color=[0, 255, 0]),
        10: dict(link=('Left_Elbow', 'Left_Paw'), id=10, color=[0, 255, 0]),
        11: dict(link=('Right_Elbow', 'Right_Paw'), id=11, color=[255, 128, 0]),
        12: dict(link=('Left_Hip', 'Left_Knee'), id=12, color=[255, 128, 0]),
        13: dict(link=('Right_Hip', 'Right_Knee'), id=13, color=[255, 128, 0]),
        14: dict(link=('Left_Knee', 'Left_Foot'), id=14, color=[0, 255, 0]),
        15: dict(link=('Right_Knee', 'Right_Foot'), id=15, color=[0, 255, 0]),
        16: dict(link=('Left_Hip', 'Tail'), id=16, color=[255, 128, 0]),
        17: dict(link=('Right_Hip', 'Tail'), id=17, color=[255, 128, 0]),
        18: dict(link=('Tail', 'Center'), id=18, color=[0, 255, 0]),
    },
    joint_weights=[
        1.5, #0
        1., #1
        1., #2
        1.2, #3
        1.2, #4
        1., #5
        1., #6
        1.2, #7
        1.2, #8
        1.5, #9
        1.5, #10
        1., #11
        1., #12
        1.2, #13
        1.2, #14
        1.5, #15
        1.5, #16
        1., #17
        1. #18
    ],

    # Note: The original paper did not provide enough information about
    # the sigmas. We modified from 'https://github.com/cocodataset/'
    # 'cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L523'
    sigmas=[
        0.026, #0 
        0.025, #1
        0.025, #2
        0.035, #3
        0.035, #4
        0.107, #5
        0.107, #6
        0.087, #7
        0.087, #8
        0.089, #9
        0.089, #10
        0.107, #11
        0.107, #12
        0.087, #13
        0.087, #14
        0.089, #15
        0.089, #16
        0.10, #17
        0.10 #18
    ])
