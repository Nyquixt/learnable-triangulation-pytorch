'''
    This file is used to generate the .NPY label file from OpenSim ground truth data
    TODO: Explain more about the data format here
'''

import numpy as np
from xml.dom.minidom import parse
import math
import cv2
import mvn.utils.multiview as multiview
from bs4 import BeautifulSoup
'''
Joint Locations
IDs:
0 pelvis/root
1 r_hip
2 r_knee
3 r_ankle
4 l_hip
5 l_knee
6 l_ankle
7 back
8 r_shoulder
9 r_elbow
10 r_wrist
11 l_shoulder
12 l_elbow
13 l_wrist

Joint Angles/Rotation/Kinematics
- contains 42 degrees of freedoms (euler) with rotations of end effectors (wrist and ankle) being 0's
'''

def cal_R(ORIENTATION): # rotation matrix
    x,y,z,w = ORIENTATION
    R = np.zeros((3,3))
    R[0,0] = w**2 + x**2 - y**2 - z**2
    R[0,1] = 2 * x * y - 2 * w * z
    R[0,2] = 2 * x * z + 2 * w * y
    R[1,0] = 2 * x * y + 2 * w * z
    R[1,1] = w**2 - x**2 + y**2 - z**2
    R[1,2] = 2 * y * z - 2 * w * x
    R[2,0] = 2 * x * z - 2 * w * y
    R[2,1] = 2 * y * z + 2 * w * x
    R[2,2] = w**2 - x**2 - y**2 + z**2
    return R

def cal_camera_param(camera):
    shot_camera = {}

    # extrinsic
    position = camera['POSITION']
    shot_camera['t'] = np.array(position).T
    orientation = camera['ORIENTATION']
    shot_camera['R'] = cal_R(orientation)
    shot_camera['t'] = -shot_camera['R'] @ shot_camera['t']
    principal_point = camera['PRINCIPAL_POINT']
    # intrinsic
    shot_camera['K'] = np.array([[float(camera['FOCAL_LENGTH']),0,float(principal_point[0])],
                                 [0,float(camera['FOCAL_LENGTH']),float(principal_point[1])],
                                 [0,        0,      1]]) # projection matrix
    shot_camera['dist'] = np.array([0, 0, 0, 0, 0]) # distortion
    
    dt = np.dtype([('R', np.float32, (3,3)),('t', np.float32, (3,)),('K', np.float32, (3,3)),('dist', np.float32, (5,))])
    param = np.array([
                (
                    tuple(shot_camera['R'].tolist()),
                    tuple(shot_camera['t'].tolist()),
                    tuple(shot_camera['K'].tolist()),
                    tuple(shot_camera['dist'].tolist())
                )
            ], dtype=dt)
    return param[0]
   
    
def read_xcp(path, cam_n): # read camera configs from file .xcp
    data = {}
    domTree = parse(path)
    rootNode = domTree.documentElement

    cameras = rootNode.getElementsByTagName("Camera")
    camera = cameras[cam_n]
    keyframes = camera.getElementsByTagName("KeyFrames")[0]
    keyframe = keyframes.getElementsByTagName("KeyFrame")[0]

    data["FOCAL_LENGTH"] = float(keyframe.getAttribute("FOCAL_LENGTH"))
    data["ORIENTATION"] = [float(i) for i in keyframe.getAttribute("ORIENTATION").split(' ')]
    data["POSITION"] = [float(i) for i in keyframe.getAttribute("POSITION").split(' ')]
    data["PRINCIPAL_POINT"] = [float(i) for i in keyframe.getAttribute("PRINCIPAL_POINT").split(' ')]
    return data

def add_offset_to_intrinsic_matrix(param):
    param[2][0][2] -= 320
    param[2][1][2] -= 180
    return param

def load_camera_param(sub, cam, path):
    data = read_xcp(path, -3 + cam)
    data = cal_camera_param(data)
    data = add_offset_to_intrinsic_matrix(data)
    return data

def load_names(line):
    line = line.split('\n')[0]
    line = line.split('\t')[2:]
    names = []
    for name in line:
        if name != '':
            names.append(name)
    return names

def load_trc(path):
    f = open(path, 'r')
    data = f.readlines()
    dst_data = []
    for i in range(len(data)):
        if i == 3:
            names = load_names(data[i])
        if i < 5:
            continue
        line = data[i].split('\t')
        line_data = []
        for j in range(2, len(line), 3): # step 3 steps for 3D coordinates
            line_data.append(np.array([float(line[j]), float(line[j + 1]), float(line[j + 2])]))
        dst_data.append(line_data)
    return dst_data, names

def load_STO_locations(sto_path):
    '''
        sto_path: path to .sto files that contain the synthetic 3D joint locations from Inverse Kinematics
        names: name of the markers in .trc files
        Note: locations in .sto files need to be multiplied by 1000
    '''
    with open(sto_path, "r") as sto_file:
        lines = sto_file.readlines()
        data = lines[5:]
        
    npy_data = []
    for frame_idx in range(len(data)):
        frame = data[frame_idx].split()
        instances = []
        for it in frame:
            instances.append(np.asarray([np.float32(x) for x in it.split(",")]))
        instances = np.array(instances)
        joints = []
        joints.append(instances[14]*1000) # pelvis
        joints.append(instances[1]*1000) # r_hip
        joints.append(instances[3]*1000) # r_knee
        joints.append(instances[5]*1000) # r_ankle
        joints.append(instances[2]*1000) # l_hip
        joints.append(instances[4]*1000) # l_knee
        joints.append(instances[6]*1000) # l_ankle
        joints.append(instances[13]*1000) # back
        joints.append(instances[7]*1000) # r_shoulder
        joints.append(instances[9]*1000) # r_elbow
        joints.append(instances[12]*1000) # r_wrist
        joints.append(instances[8]*1000) # l_shoulder
        joints.append(instances[10]*1000) # l_elbow
        joints.append(instances[11]*1000) # l_wrist
        npy_data.append(joints)
    npy_data = np.array(npy_data)
    return npy_data

def load_MOT_angles(mot_path):
    '''
        mot_path: .mot files recording joint angles
        output: rotations in quaternions
    '''
    # indices of the angles according to the .mot files; NOTE: angles are in degrees; order is ZXY
    angles_indices = [[4, 5, 6], # pelvis_translation
                    [1, 2, 3], # pelvis_rotation
                    [7, 8, 9], # hip_r
                    [10, -1, -1], # knee_r
                    -1, # ankle_r
                    [14, 15, 16], # hip_l
                    [17, -1, -1], # knee_l
                    -1, # ankle_l
                    [21, 22, 23], # back
                    [24, 25, 26], # arm_r
                    [27, -1, -1], # elbow_r
                    -1, #wrist_r
                    [31, 32, 33], # arm_l
                    [34, -1, -1], #  # elbow_l
                    -1] # wrist_l
    mot_file = open(mot_path, 'r')
    mot_frames = mot_file.readlines()[11:] # each line represents each frame
    mot_file.close()

    rotations_by_frames = []
    translations_by_frames = []
    for line in mot_frames:
        frame = line.split()
        translation = [float(frame[angles_indices[0][0]]), float(frame[angles_indices[0][1]]), float(frame[angles_indices[0][2]])]
        translations_by_frames.append(translation)
        rotations = []
        for an in angles_indices[1:]:
            if type(an) == list:
                if an[0] == -1:
                    rotations.append([0., 0., float(frame[an[2]])])
                else:
                    rotations.append([float(frame[an[0]]), float(frame[an[1]]), float(frame[an[2]])])
            elif an == -1:
                rotations.append([0., 0., 0.])
        rotations_by_frames.append(rotations)

    translations_by_frames = np.array(translations_by_frames, dtype=np.float64) * 1000 # m to mm
    rotations_by_frames = np.array(rotations_by_frames, dtype=np.float32)

    # handle hip_l, shoulder_l, elbow_r, elbow_l rotation axes due to some subtle details in .osim model
    rotations_by_frames[:, 4, 1:] *= -1
    rotations_by_frames[:, 11, 1:] *= -1

    rotations_by_frames[:, 9, 0] *= 0.97386183000000004
    rotations_by_frames[:, 9, 1] += 4 * np.arcsin(0.022269)

    rotations_by_frames[:, 12, 0] *= 0.97386183000000004
    rotations_by_frames[:, 12, 1] += 4 * np.arcsin(0.022269)
    
    return translations_by_frames, rotations_by_frames

def get_coordinates_from_npy_file(subject_idx=0, action_idx=0, frame_idx=0):
    data = np.load("/data/kien/Roof_Phase3/annotations/roofing-multiview-v2.npy", allow_pickle=True).item()
    ground_truth = data["table"]
    frames = []
    for fr in ground_truth:
        if fr[0] == subject_idx and fr[1] == action_idx:
            frames.append(fr)
    
    coords = frames[frame_idx][3]
    return coords

def load_model_offsets(osim_file, subject_idx):
    with open(osim_file, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, "xml")

    joint_names = ["hip_r", "hip_l", "acromial_r", "acromial_l", "elbow_r", "elbow_l", "back", "ankle_r", "ankle_l", "radius_hand_r", "radius_hand_l"]
    parents = ["pelvis", "pelvis", "torso", "torso", "humerus_r", "humerus_l", "pelvis", "tibia_r", "tibia_l", "radius_r", "radius_l"]

    offsets = {}
    offsets['pelvis'] = [0., 0., 0.]
    # calculate knee offset because .osim doesn't provide this data for some reasons
    coords = get_coordinates_from_npy_file(subject_idx, 0, 0)
    knee_r = np.linalg.norm(coords[1] - coords[2]) / 1000
    knee_l = np.linalg.norm(coords[4] - coords[3]) / 1000
    offset_knee = (knee_r + knee_l) / 2
    offsets['knee_r'] = [0., -offset_knee, 0.]
    offsets['knee_l'] = [0., -offset_knee, 0.]

    for idx, name in enumerate(joint_names):
        joint = bs_data.find('CustomJoint', {'name': name})
        offset = joint.find('frames').find('PhysicalOffsetFrame', {'name': f'{parents[idx]}_offset'}).find("translation")
        offsets[name] = [float(x) for x in offset.text.strip().split(" ")]
    
    offsets_list = [
        offsets['pelvis'], #0
        offsets['hip_r'],  #1
        offsets['knee_r'], #2
        offsets['ankle_r'], #3
        offsets['hip_l'],  #4
        offsets['knee_l'],  #5
        offsets['ankle_l'], #6
        offsets['back'],   #7
        offsets['acromial_r'], #8
        offsets['elbow_r'], #9
        offsets['radius_hand_r'], #10
        offsets['acromial_l'], #11
        offsets['elbow_l'], #12
        offsets['radius_hand_l'] #13
    ]
    offsets_list = np.array(offsets_list, dtype=np.float64) * 1000 # m to mm
    # offsets_list = [list(x) for x in offsets_list]
    return offsets_list

def cal_bbox(points):
    t = 99999
    l = 99999
    b = 0
    r = 0
    for p in range(points.shape[0] - 1):
        t = min(t, points[p][1])
        l = min(l, points[p][0])
        b = max(b, points[p][1])
        r = max(r, points[p][0])
    t = max(t - 100,0)
    l = max(l - 100,0)
    b = min(b + 100, 1080 - 1)
    r = min(r + 100, 1920 - 1)
    return int(t), int(l), math.ceil(b), math.ceil(r)

def cal_person_tlbr_bbox(sto_data, cam_param):
    all_bbox = []
    for f in range(sto_data.shape[0]):
        frame_bbox = []
        for c in range(3):
            retval_camera = multiview.Camera(cam_param[c][0], cam_param[c][1], cam_param[c][2], cam_param[c][3])
            keypoints_2d = multiview.project_3d_points_to_image_plane_without_distortion(retval_camera.projection, sto_data[f])
            t,l,b,r = cal_bbox(keypoints_2d)
            frame_bbox.append([t,l,b,r])
        all_bbox.append(frame_bbox)
    all_bbox = np.array(all_bbox)

    return all_bbox

def generate_table(coords_sto_path, angles_mot_path, sub, act, cam_param, error_frame):
    keypoints_3d = load_STO_locations(coords_sto_path)
    translations, rotations = load_MOT_angles(angles_mot_path)
    bbox_tlbr = cal_person_tlbr_bbox(keypoints_3d, cam_param)

    print(keypoints_3d.shape, translations.shape)

    table_data = []
    # num frames between keypoints and angles might mismatch
    for frame in range(min(keypoints_3d.shape[0], translations.shape[0])):
        if frame in error_frame:
            continue
        dt = np.dtype([('subject_idx', np.int32, (1)),
                       ('action_idx', np.int32, (1)),
                       ('frame_idx', np.int32, (1)),
                       ('keypoints', np.float32, (14, 3)),
                       ('translations', np.float32, (3,)),
                       ('rotations', np.float32, (14, 3)),
                       ('bbox_by_camera_tlbr', np.int32, (3, 4))
                    ])
        # main data format of npy['table']
        table = np.array([
                            (
                                (sub), 
                                (act), 
                                (frame),
                                tuple(keypoints_3d[frame].tolist()),
                                tuple(translations[frame].tolist()),
                                tuple(rotations[frame].tolist()),
                                tuple(bbox_tlbr[frame].tolist())
                            )
                        ], dtype=dt)
        table_data.append(table[0])
    return table_data

def check_video(path1, path2, path3):
    # check videos to find the error frames. we will ignore error frames in table/data generation. 
    error_frame = []
    paths = [path1, path2, path3]
    
    for p in paths:
        vc = cv2.VideoCapture(p)
        success, frame = vc.read()
        frame_idx = 0
        while success:
            mean = np.mean(frame)
            if mean < 10:
                if frame_idx not in error_frame:
                    error_frame.append(frame_idx)
            success, frame = vc.read()
            frame_idx += 1
        vc.release()
    
    return error_frame

def main():
    save_path = "roofing-multiview-fk"
    src_data_path = "/data/kien/Roof_Phase3/"
    coords_sto_path = "/data/kien/Roof_Phase3/keypoint_gt/"
    angles_mot_path = "/data/kien/Roof_Phase3/angle_gt/"
    osim_file_path = "/data/kien/Roof_Phase3/osim_file/"

    data = {}
    data['subject_names'] = ['S04','S05','S06','S07','S08','S09','S10']
    data['camera_names'] = ['Cam0','Cam1','Cam2']
    data['action_names'] = ['Act0','Act1','Act2']
    data['cameras'] = \
    [
        [[],[],[]],
        [[],[],[]],
        [[],[],[]],
        [[],[],[]],
        [[],[],[]],
        [[],[],[]],
        [[],[],[]]
    ]
    data['offsets'] = []
    data['table'] = []
    
    sub = 0
    for act in range(3):
        if act == 0:
            path1 = src_data_path + "Subject04_10-03-2019/Trial 27.2127642.20191210102736.avi"
            path2 = src_data_path + "Subject04_10-03-2019/Trial 27.2127898.20191210102736.avi"
            path3 = src_data_path + "Subject04_10-03-2019/Trial 27.2128267.20191210102736.avi"
            error_frame = check_video(path1, path2, path3)
        if act == 1:
            path1 = src_data_path + "Subject04_10-03-2019/Trial 28.2127642.20191210104316.avi"
            path2 = src_data_path + "Subject04_10-03-2019/Trial 28.2127898.20191210104316.avi"
            path3 = src_data_path + "Subject04_10-03-2019/Trial 28.2128267.20191210104316.avi"
            error_frame = check_video(path1, path2, path3)
        if act == 2:
            path1 = src_data_path + "Subject04_10-03-2019/Trial 29.2127642.20191210104426.avi"
            path2 = src_data_path + "Subject04_10-03-2019/Trial 29.2127898.20191210104426.avi"
            path3 = src_data_path + "Subject04_10-03-2019/Trial 29.2128267.20191210104426.avi"
            error_frame = check_video(path1, path2, path3)
        for cam in range(3):
            data['cameras'][sub][cam] = load_camera_param(sub, cam, src_data_path + "Subject04_10-03-2019/Trial 2" + str(act + 7) + ".xcp")
        
        data['table'] += generate_table(coords_sto_path + "S04_" + str(act + 1) + "_OutputsVec3.sto",
                                        angles_mot_path + "S04_" + str(act + 1) + "_IK_Result.mot",
                                        sub, act, data['cameras'][sub], error_frame)
    offsets = load_model_offsets(osim_file_path + "model_S04_scaled.osim", sub)
    data['offsets'].append(offsets)

    sub = 1
    for act in range(3):
        if act == 0:
            path1 = src_data_path + "Subject05_10-07-2019/Trial 11.2127642.20191210110039.avi"
            path2 = src_data_path + "Subject05_10-07-2019/Trial 11.2127898.20191210110039.avi"
            path3 = src_data_path + "Subject05_10-07-2019/Trial 11.2128267.20191210110039.avi"
            error_frame = check_video(path1, path2, path3)
        if act == 1:
            path1 = src_data_path + "Subject05_10-07-2019/Trial 12.2127642.20191210110200.avi"
            path2 = src_data_path + "Subject05_10-07-2019/Trial 12.2127898.20191210110200.avi"
            path3 = src_data_path + "Subject05_10-07-2019/Trial 12.2128267.20191210110200.avi"
            error_frame = check_video(path1, path2, path3)
        if act == 2:
            path1 = src_data_path + "Subject05_10-07-2019/Trial 13.2127642.20191210110318.avi"
            path2 = src_data_path + "Subject05_10-07-2019/Trial 13.2127898.20191210110318.avi"
            path3 = src_data_path + "Subject05_10-07-2019/Trial 13.2128267.20191210110318.avi"
            error_frame = check_video(path1, path2, path3)
        for cam in range(3):
            data['cameras'][sub][cam] = load_camera_param(sub, cam, src_data_path + "Subject05_10-07-2019/Trial 1" + str(act + 1) + ".xcp")
        
        data['table'] += generate_table(coords_sto_path + "S05_" + str(act + 1) + "_OutputsVec3.sto",
                                        angles_mot_path + "S05_" + str(act + 1) + "_IK_Result.mot",
                                        sub, act, data['cameras'][sub], error_frame)
    offsets = load_model_offsets(osim_file_path + "model_S05_scaled.osim", sub)
    data['offsets'].append(offsets)
    
    sub = 2
    for act in range(3):
        if act == 0:
            path1 = src_data_path + "Subject06_10-08-2019/Trial 21.2127642.20191210121034.avi"
            path2 = src_data_path + "Subject06_10-08-2019/Trial 21.2127898.20191210121034.avi"
            path3 = src_data_path + "Subject06_10-08-2019/Trial 21.2128267.20191210121034.avi"
            error_frame = check_video(path1, path2, path3)
        if act == 1:
            path1 = src_data_path + "Subject06_10-08-2019/Trial 22.2127642.20191210121204.avi"
            path2 = src_data_path + "Subject06_10-08-2019/Trial 22.2127898.20191210121204.avi"
            path3 = src_data_path + "Subject06_10-08-2019/Trial 22.2128267.20191210121204.avi"
            error_frame = check_video(path1, path2, path3)
        if act == 2:
            path1 = src_data_path + "Subject06_10-08-2019/Trial 23.2127642.20191210121340.avi"
            path2 = src_data_path + "Subject06_10-08-2019/Trial 23.2127898.20191210121340.avi"
            path3 = src_data_path + "Subject06_10-08-2019/Trial 23.2128267.20191210121340.avi"
            error_frame = check_video(path1, path2, path3)
        for cam in range(3):
            data['cameras'][sub][cam] = load_camera_param(sub, cam, src_data_path + "Subject06_10-08-2019/Trial 2" + str(act + 1) + ".xcp")
        
        data['table'] += generate_table(coords_sto_path + "S06_" + str(act + 1) + "_OutputsVec3.sto",
                                        angles_mot_path + "S06_" + str(act + 1) + "_IK_Result.mot",
                                        sub, act, data['cameras'][sub], error_frame)
    
    offsets = load_model_offsets(osim_file_path + "model_S06_scaled.osim", sub)
    data['offsets'].append(offsets)

    sub = 3
    for act in range(3):
        if act == 0:
            path1 = src_data_path + "Subject07_10-09-2019/Trial 21.2127642.20191210122130.avi"
            path2 = src_data_path + "Subject07_10-09-2019/Trial 21.2127898.20191210122130.avi"
            path3 = src_data_path + "Subject07_10-09-2019/Trial 21.2128267.20191210122130.avi"
            error_frame = check_video(path1, path2, path3)
        if act == 1:
            path1 = src_data_path + "Subject07_10-09-2019/Trial 22.2127642.20191210122259.avi"
            path2 = src_data_path + "Subject07_10-09-2019/Trial 22.2127898.20191210122259.avi"
            path3 = src_data_path + "Subject07_10-09-2019/Trial 22.2128267.20191210122259.avi"
            error_frame = check_video(path1, path2, path3)
        if act == 2:
            path1 = src_data_path + "Subject07_10-09-2019/Trial 23.2127642.20191210122422.avi"
            path2 = src_data_path + "Subject07_10-09-2019/Trial 23.2127898.20191210122422.avi"
            path3 = src_data_path + "Subject07_10-09-2019/Trial 23.2128267.20191210122422.avi"
            error_frame = check_video(path1, path2, path3)
        for cam in range(3):
            data['cameras'][sub][cam] = load_camera_param(sub, cam, src_data_path + "Subject07_10-09-2019/Trial 2" + str(act + 1) + ".xcp")
        
        data['table'] += generate_table(coords_sto_path + "S07_" + str(act + 1) + "_OutputsVec3.sto",
                                        angles_mot_path + "S07_" + str(act + 1) + "_IK_Result.mot",
                                        sub, act, data['cameras'][sub], error_frame)
    
    offsets = load_model_offsets(osim_file_path + "model_S07_scaled.osim", sub)
    data['offsets'].append(offsets)

    sub = 4
    for act in range(3):
        if act == 0:
            path1 = src_data_path + "Subject08_10-09-2019/Trial 31.2127642.20191016145653.avi"
            path2 = src_data_path + "Subject08_10-09-2019/Trial 31.2127898.20191016145653.avi"
            path3 = src_data_path + "Subject08_10-09-2019/Trial 31.2128267.20191016145653.avi"
            error_frame = check_video(path1, path2, path3)
        if act == 1:
            path1 = src_data_path + "Subject08_10-09-2019/Trial 32.2127642.20191016145831.avi"
            path2 = src_data_path + "Subject08_10-09-2019/Trial 32.2127898.20191016145831.avi"
            path3 = src_data_path + "Subject08_10-09-2019/Trial 32.2128267.20191016145831.avi"
            error_frame = check_video(path1, path2, path3)
        if act == 2:
            path1 = src_data_path + "Subject08_10-09-2019/Trial 33.2127642.20191016150015.avi"
            path2 = src_data_path + "Subject08_10-09-2019/Trial 33.2127898.20191016150015.avi"
            path3 = src_data_path + "Subject08_10-09-2019/Trial 33.2128267.20191016150015.avi"
            error_frame = check_video(path1, path2, path3)
        for cam in range(3):
            data['cameras'][sub][cam] = load_camera_param(sub, cam, src_data_path + "Subject08_10-09-2019/Trial 3" + str(act + 1) + ".xcp")
        
        data['table'] += generate_table(coords_sto_path + "S08_" + str(act + 1) + "_OutputsVec3.sto",
                                        angles_mot_path + "S08_" + str(act + 1) + "_IK_Result.mot",
                                        sub, act, data['cameras'][sub], error_frame)
    offsets = load_model_offsets(osim_file_path + "model_S08_scaled.osim", sub)
    data['offsets'].append(offsets)

    sub = 5
    for act in range(3):
        if act == 0:
            path1 = src_data_path + "Subject09_10-11-2019/Trial 11.2127642.20191016142806.avi"
            path2 = src_data_path + "Subject09_10-11-2019/Trial 11.2127898.20191016142806.avi"
            path3 = src_data_path + "Subject09_10-11-2019/Trial 11.2128267.20191016142806.avi"
            error_frame = check_video(path1, path2, path3)
        if act == 1:
            path1 = src_data_path + "Subject09_10-11-2019/Trial 12.2127642.20191016142937.avi"
            path2 = src_data_path + "Subject09_10-11-2019/Trial 12.2127898.20191016142937.avi"
            path3 = src_data_path + "Subject09_10-11-2019/Trial 12.2128267.20191016142937.avi"
            error_frame = check_video(path1, path2, path3)
        if act == 2:
            path1 = src_data_path + "Subject09_10-11-2019/Trial 13.2127642.20191016143114.avi"
            path2 = src_data_path + "Subject09_10-11-2019/Trial 13.2127898.20191016143114.avi"
            path3 = src_data_path + "Subject09_10-11-2019/Trial 13.2128267.20191016143114.avi"
            error_frame = check_video(path1, path2, path3)
        for cam in range(3):
            data['cameras'][sub][cam] = load_camera_param(sub, cam, src_data_path + "Subject09_10-11-2019/Trial 1" + str(act + 1) + ".xcp")
        
        data['table'] += generate_table(coords_sto_path + "S09_" + str(act + 1) + "_OutputsVec3.sto",
                                        angles_mot_path + "S09_" + str(act + 1) + "_IK_Result.mot",
                                        sub, act, data['cameras'][sub], error_frame)
    
    offsets = load_model_offsets(osim_file_path + "model_S09_scaled.osim", sub)
    data['offsets'].append(offsets)

    sub = 6
    for act in range(3):
        if act == 0:
            path1 = src_data_path + "Subject10_10-16-2019/Trial 31.2127642.20191016135245.avi"
            path2 = src_data_path + "Subject10_10-16-2019/Trial 31.2127898.20191016135245.avi"
            path3 = src_data_path + "Subject10_10-16-2019/Trial 31.2128267.20191016135245.avi"
            error_frame = check_video(path1, path2, path3)
        if act == 1:
            path1 = src_data_path + "Subject10_10-16-2019/Trial 32.2127642.20191016135430.avi"
            path2 = src_data_path + "Subject10_10-16-2019/Trial 32.2127898.20191016135430.avi"
            path3 = src_data_path + "Subject10_10-16-2019/Trial 32.2128267.20191016135430.avi"
            error_frame = check_video(path1, path2, path3)
        if act == 2:
            path1 = src_data_path + "Subject10_10-16-2019/Trial 33.2127642.20191016135549.avi"
            path2 = src_data_path + "Subject10_10-16-2019/Trial 33.2127898.20191016135549.avi"
            path3 = src_data_path + "Subject10_10-16-2019/Trial 33.2128267.20191016135549.avi"
            error_frame = check_video(path1, path2, path3)
        for cam in range(3):
            data['cameras'][sub][cam] = load_camera_param(sub, cam , src_data_path + "Subject10_10-16-2019/Trial 3" + str(act + 1) + ".xcp")
        
        data['table'] += generate_table(coords_sto_path + "S10_" + str(act + 1) + "_OutputsVec3.sto",
                                        angles_mot_path + "S10_" + str(act + 1) + "_IK_Result.mot",
                                        sub, act , data['cameras'][sub], error_frame)
    
    offsets = load_model_offsets(osim_file_path + "model_S10_scaled.osim", sub)
    data['offsets'].append(offsets)

    data['cameras'] = np.array(data['cameras'])
    data['offsets'] = np.array(data['offsets'])
    data['table'] = np.array(data['table'])
    print(f"Number of frame: {data['table'].shape[0]}")
    
    np.save(save_path, data)

if __name__ == '__main__':
    main()
    print("success")