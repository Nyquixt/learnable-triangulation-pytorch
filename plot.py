import pickle
import numpy as np
import matplotlib.pyplot as plt
from mvn.utils.op import translate_quaternion_to_euler

angle_names = [
    'knee_angle_r',
    'hip_flexion_r',
    'hip_adduction_r',
    'hip_rotation_r',
    'hip_flexion_l',
    'hip_adduction_l',
    'hip_rotation_l',
    'knee_angle_l',
    'elbow_flexion_r',
    'shoulder_flex_r',
    'shoulder_add_r',
    'shoulder_rot_r',
    'shoulder_flex_l',
    'shoulder_add_l',
    'shoulder_rot_l',
    'elbow_flexion_l'
]

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def main(gt_angles_trajs, pred_angles_trajs, exp_name, smoothness):
    if pred_angles_trajs.shape[1] == 32:
        eulers = []
        for a in pred_angles_trajs:
            eulers.append(translate_quaternion_to_euler(list(a)))
        pred_angles_trajs = np.array(eulers)
    # S08's end frames for Act0, 1, 2
    S08 = [0, 1600, 3320, 4874]
    S08_pred_angles_trajs = pred_angles_trajs[:4874]
    S08_gt_angles_trajs = gt_angles_trajs[:4874]

    # S10's end frames for Act0, 1, 2
    S10 = [0, 1170, 2305, 3443]
    S10_pred_angles_trajs = pred_angles_trajs[4874:]
    S10_gt_angles_trajs = gt_angles_trajs[4874:]

    for i in range(1, 4):
        preds_frames = S08_pred_angles_trajs[S08[i-1] : S08[i]]
        gt_frames = S08_gt_angles_trajs[S08[i-1] : S08[i]]
        
        fig, axes =  plt.subplots(4, 4, figsize=(18, 10))
        for r in range(4):
            for c in range(4):
                angle = r * 4  + c
                if r < 3:
                    axes[r][c].get_xaxis().set_visible(False)
                axes[r][c].set_title(angle_names[angle])
                axes[r][c].plot(smooth(preds_frames[:, angle], smoothness), color='blue')
                axes[r][c].plot(gt_frames[:, angle], color='red')
        fig.legend(['predicted', 'groundtruth'])
        plt.savefig("{}_S08_{}.pdf".format(exp_name, i))

    for i in range(1, 4):
        preds_frames = S10_pred_angles_trajs[S10[i-1] : S10[i]]
        gt_frames = S10_gt_angles_trajs[S10[i-1] : S10[i]]
        
        fig, axes =  plt.subplots(4, 4, figsize=(18, 10))
        for r in range(4):
            for c in range(4):
                angle = r * 4  + c
                if r < 3:
                    axes[r][c].get_xaxis().set_visible(False)
                axes[r][c].set_title(angle_names[angle])
                axes[r][c].plot(smooth(preds_frames[:, angle], smoothness), color='blue')
                axes[r][c].plot(gt_frames[:, angle], color='red')
        fig.legend(['predicted', 'groundtruth'])
        plt.savefig("{}_S10_{}.pdf".format(exp_name, i))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Save predictions for easier analysis pipeline')
    parser.add_argument('--records', '-r', required=True, type=str, help='prediction records file')
    parser.add_argument('--outfile', '-o', required=True, type=str, help='output file name')
    parser.add_argument('--smooth', '-s', default=0.5, type=float, help='smoothness of plot')
    args = parser.parse_args()

    labels = np.load('../baseline-angles/roofing-multiview-v2.npy', allow_pickle=True).item()

    test_subjects = ['S08', 'S10']
    test_subjects  = list(labels['subject_names'].index(x) for x in test_subjects)

    indices = []

    mask = np.isin(labels['table']['subject_idx'], test_subjects, assume_unique=True)
    indices.append(np.nonzero(mask)[0][::1])

    labels['table'] = labels['table'][np.concatenate(indices)]

    angles_gt = np.deg2rad(labels['table']['angles'][:, :16])

    with open(args.records, 'rb') as infile:
        data = pickle.load(infile)
    angles_pred = data['angles']

    main(angles_gt, angles_pred, args.outfile, args.smooth)