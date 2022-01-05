import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main(gt_angles_trajs, pred_angles_trajs):

    mae = mean_absolute_error(gt_angles_trajs, pred_angles_trajs)
    mse = mean_squared_error(gt_angles_trajs, pred_angles_trajs)
    rmse = np.sqrt(mse)
    r2 = r2_score(gt_angles_trajs, pred_angles_trajs)

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
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