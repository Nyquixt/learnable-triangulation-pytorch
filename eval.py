import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

def main(gt_angles_trajs, pred_angles_trajs):
    if pred_angles_trajs.shape[1] == 32:
        eulers = []
        for a in pred_angles_trajs:
            eulers.append(translate_quaternion_to_euler(list(a)))
        pred_angles_trajs = np.array(eulers)
    print('Mean Squared Error:')

    for idx, an in enumerate(angle_names):
        mse = mean_squared_error(gt_angles_trajs[:, idx], pred_angles_trajs[:, idx])
        print(f'{an}: {mse:.3f} rad ({np.rad2deg(mse):.3f} deg)')

    print('-------------------')
    print('Average:')
    mae = mean_absolute_error(gt_angles_trajs, pred_angles_trajs)
    mse = mean_squared_error(gt_angles_trajs, pred_angles_trajs)
    r2 = r2_score(gt_angles_trajs, pred_angles_trajs)
    print(f'MAE: {mae} rad ({np.rad2deg(mae):.3f} deg)')
    print(f'MSE: {mse} rad ({np.rad2deg(mse):.3f} deg)')
    print(f'R2: {r2}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Save predictions for easier analysis pipeline')
    parser.add_argument('--records', '-r', required=True, type=str, help='prediction records file')
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

    main(angles_gt, angles_pred)