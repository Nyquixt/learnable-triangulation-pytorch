import os
from collections import defaultdict

import numpy as np
import cv2
from torch.utils.data import Dataset

from mvn.utils.multiview import Camera
from mvn.utils.img import resize_image, crop_image, normalize_image, scale_bbox
from scipy.spatial.transform import Rotation as R

class RoofingMultiViewDataset(Dataset):
    """
        Human3.6M for multiview tasks.
    """
    def __init__(self,
                 roofing_root='/data/kien/Roof_Phase3/images',
                 labels_path='/data/kien/Roof_Phase3/annotations/roofing-multiview-fk.npy',
                 pred_results_path=None,
                 image_shape=(256, 256),
                 train=False,
                 test=False,
                 retain_every_n_frames_in_test=1,
                 cuboid_side=2000.0,
                 scale_bbox=1.5,
                 norm_image=True,
                 kind="mpii",
                 ignore_cameras=[],
                 crop=True,
                 heatmap_res = 96
                 ):
        """
            roofing_root:
                Path to 'processed/' directory in Human3.6M
            labels_path:
                Path to 'human36m-multiview-labels.npy' generated by 'generate-labels-npy-multiview.py'
                from https://github.sec.samsung.net/RRU8-VIOLET/human36m-preprocessing
            retain_every_n_frames_in_test:
                By default, there are 159 181 frames in training set and 26 634 in test (val) set.
                With this parameter, test set frames will be evenly skipped frames so that the
                test set size is `26634 // retain_every_n_frames_test`.
                Use a value of 13 to get 2049 frames in test set.
            kind:
                Keypoint format, 'mpii' or 'human36m'
            ignore_cameras:
                A list with indices of cameras to exclude (0 to 3 inclusive)
        """
        assert train or test, '`Human36MMultiViewDataset` must be constructed with at least ' \
                              'one of `test=True` / `train=True`'
        assert kind in ("mpii", "human36m")

        self.roofing_root = roofing_root
        self.labels_path = labels_path
        self.image_shape = None if image_shape is None else tuple(image_shape)
        self.scale_bbox = scale_bbox
        self.norm_image = norm_image
        self.cuboid_side = cuboid_side
        self.kind = kind
        self.ignore_cameras = ignore_cameras
        self.crop = crop
        self.heatmap_res = heatmap_res

        self.labels = np.load(labels_path, allow_pickle=True).item()

        n_cameras = len(self.labels['camera_names'])
        assert all(camera_idx in range(n_cameras) for camera_idx in self.ignore_cameras)

        train_subjects = ['S04', 'S05', 'S06', 'S07', 'S09']
        test_subjects = ['S08', 'S10']

        train_subjects = list(self.labels['subject_names'].index(x) for x in train_subjects)
        test_subjects  = list(self.labels['subject_names'].index(x) for x in test_subjects)

        indices = []
        if train:
            mask = np.isin(self.labels['table']['subject_idx'], train_subjects, assume_unique=True)
            indices.append(np.nonzero(mask)[0])
        if test:
            mask = np.isin(self.labels['table']['subject_idx'], test_subjects, assume_unique=True)
            indices.append(np.nonzero(mask)[0][::retain_every_n_frames_in_test])

        self.labels['table'] = self.labels['table'][np.concatenate(indices)]

        self.num_keypoints = 14
        assert self.labels['table']['keypoints'].shape[1] == 14, "Use a newer 'labels' file"

        self.keypoints_3d_pred = None
        if pred_results_path is not None:
            pred_results = np.load(pred_results_path, allow_pickle=True)
            keypoints_3d_pred = pred_results['keypoints_3d'][np.argsort(pred_results['indexes'])]
            self.keypoints_3d_pred = keypoints_3d_pred[::retain_every_n_frames_in_test]
            assert len(self.keypoints_3d_pred) == len(self), \
                f"[train={train}, test={test}] {labels_path} has {len(self)} samples, but '{pred_results_path}' " + \
                f"has {len(self.keypoints_3d_pred)}. Did you follow all preprocessing instructions carefully?"

    def __len__(self):
        return len(self.labels['table'])

    def __getitem__(self, idx):
        sample = defaultdict(list) # return value
        shot = self.labels['table'][idx]

        subject = self.labels['subject_names'][shot['subject_idx']]
        action = self.labels['action_names'][shot['action_idx']]
        offsets = self.labels['offsets'][shot['subject_idx']] # retrieve offset for each individuals
        frame_idx = shot['frame_idx']

        for camera_idx, camera_name in enumerate(self.labels['camera_names']):
            if camera_idx in self.ignore_cameras:
                continue

            # load bounding box
            bbox = shot['bbox_by_camera_tlbr'][camera_idx][[1,0,3,2]] # TLBR to LTRB
            bbox_height = bbox[2] - bbox[0]
            if bbox_height == 0:
                # convention: if the bbox is empty, then this view is missing
                continue

            # scale the bounding box
            bbox = scale_bbox(bbox, self.scale_bbox)

            # load image
            image_path = os.path.join(
                self.roofing_root, subject, action, camera_name, '{}.jpg'.format(frame_idx))
            assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
            image = cv2.imread(image_path)

            # load camera
            shot_camera = self.labels['cameras'][shot['subject_idx'], camera_idx]
            retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'], shot_camera['dist'], camera_name)

            if self.crop:
                # crop image
                image = crop_image(image, bbox)
                retval_camera.update_after_crop(bbox)

            if self.image_shape is not None:
                # resize
                image_shape_before_resize = image.shape[:2]
                image = resize_image(image, self.image_shape)
                retval_camera.update_after_resize(image_shape_before_resize, self.image_shape)

                sample['image_shapes_before_resize'].append(image_shape_before_resize)

            if self.norm_image:
                image = normalize_image(image)

            sample['images'].append(image)
            sample['detections'].append(bbox + (1.0,)) # TODO add real confidences
            sample['cameras'].append(retval_camera)
            sample['proj_matrices'].append(retval_camera.projection)

            # 2D keypoints
            # keypoints_2d = project_3d_points_to_image_plane_without_distortion(retval_camera.projection, shot['keypoints'])
            # resize once again down to heatmap resolution
            # kpt2d = keypoints_2d / self.image_shape[0] * self.heatmap_res
            # n_joints, _ = kpt2d.shape
            # kpt2d_heatmap = np.zeros((n_joints, self.heatmap_res, self.heatmap_res))
            # for i in range(n_joints):
            #     kpt2d_heatmap[i], _ = draw_labelmap(kpt2d_heatmap[i], kpt2d[i], sigma=1, type='Gaussian')

            # sample['heatmaps_2d'].append(kpt2d_heatmap)

        # 3D keypoints
        # add dummy confidences
        sample['keypoints_3d'] = np.pad(
            shot['keypoints'][:self.num_keypoints],
            ((0,0), (0,1)), 'constant', constant_values=1.0)

        # build cuboid
        # base_point = sample['keypoints_3d'][6, :3]
        # sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
        # position = base_point - sides / 2
        # sample['cuboids'] = volumetric.Cuboid3D(position, sides)

        # add ground truth angles, convert from degrees to radians
        sample['translations'] = shot['translations'] # global root position
        sample['rotations'] = shot['rotations'] # zxy euler in degrees

        # save sample's index
        sample['indexes'] = idx

        if self.keypoints_3d_pred is not None:
            sample['pred_keypoints_3d'] = self.keypoints_3d_pred[idx]

        sample.default_factory = None
        return sample

    def evaluate_using_per_pose_error(self, per_pose_error, split_by_subject):
        def evaluate_by_actions(self, per_pose_error, mask=None):
            if mask is None:
                mask = np.ones_like(per_pose_error, dtype=bool)

            action_scores = {
                'Average': {'total_loss': per_pose_error[mask].sum(), 'frame_count': np.count_nonzero(mask)}
            }

            for action_idx in range(len(self.labels['action_names'])):
                action_mask = (self.labels['table']['action_idx'] == action_idx) & mask
                action_per_pose_error = per_pose_error[action_mask]
                action_scores[self.labels['action_names'][action_idx]] = {
                    'total_loss': action_per_pose_error.sum(), 'frame_count': len(action_per_pose_error)
                }

            action_names_without_trials = \
                [name[:-2] for name in self.labels['action_names'] if name.endswith('-1')]

            for action_name_without_trial in action_names_without_trials:
                combined_score = {'total_loss': 0.0, 'frame_count': 0}

                for trial in 1, 2:
                    action_name = '%s-%d' % (action_name_without_trial, trial)
                    combined_score['total_loss' ] += action_scores[action_name]['total_loss']
                    combined_score['frame_count'] += action_scores[action_name]['frame_count']
                    del action_scores[action_name]

                action_scores[action_name_without_trial] = combined_score

            for k, v in action_scores.items():
                action_scores[k] = float('nan') if v['frame_count'] == 0 else (v['total_loss'] / v['frame_count'])

            return action_scores

        subject_scores = {
            'Average': evaluate_by_actions(self, per_pose_error)
        }

        for subject_idx in range(len(self.labels['subject_names'])):
            subject_mask = self.labels['table']['subject_idx'] == subject_idx
            subject_scores[self.labels['subject_names'][subject_idx]] = \
                evaluate_by_actions(self, per_pose_error, subject_mask)

        return subject_scores

    def evaluate(self, keypoints_3d_predicted, split_by_subject=False):
        keypoints_gt = self.labels['table']['keypoints'][:, :self.num_keypoints]
        if keypoints_3d_predicted.shape != keypoints_gt.shape:
            raise ValueError(
                '`keypoints_3d_predicted` shape should be %s, got %s' % \
                (keypoints_gt.shape, keypoints_3d_predicted.shape))

        # mean error per 14 joints in mm, for each pose
        per_pose_error = np.sqrt(((keypoints_gt - keypoints_3d_predicted) ** 2).sum(2)).mean(1)

        # relative mean error per 14 joints in mm, for each pose
        root_index = 0

        keypoints_gt_relative = keypoints_gt - keypoints_gt[:, root_index:root_index + 1, :]
        keypoints_3d_predicted_relative = keypoints_3d_predicted - keypoints_3d_predicted[:, root_index:root_index + 1, :]

        per_pose_error_relative = np.sqrt(((keypoints_gt_relative - keypoints_3d_predicted_relative) ** 2).sum(2)).mean(1)

        result = {
            'per_pose_error': self.evaluate_using_per_pose_error(per_pose_error, split_by_subject),
            'per_pose_error_relative': self.evaluate_using_per_pose_error(per_pose_error_relative, split_by_subject)
        }

        return result['per_pose_error_relative']['Average']['Average'], result

    def evaluate_angles(self, rotations_pred):
        def evaluate_by_actions(self, per_pose_error, mask=None):
            if mask is None:
                mask = np.ones_like(per_pose_error, dtype=bool)

            action_scores = {
                'Average': {'total_loss': per_pose_error[mask].sum(), 'frame_count': np.count_nonzero(mask)}
            }

            for action_idx in range(len(self.labels['action_names'])):
                action_mask = (self.labels['table']['action_idx'] == action_idx) & mask
                action_per_pose_error = per_pose_error[action_mask]
                action_scores[self.labels['action_names'][action_idx]] = {
                    'total_loss': action_per_pose_error.sum(), 'frame_count': len(action_per_pose_error)
                }

            for k, v in action_scores.items():
                action_scores[k] = float('nan') if v['frame_count'] == 0 else (v['total_loss'] / v['frame_count'])

            return action_scores
        
        rotations_gt = self.labels['table']['rotations']
        gt = []
        for r in rotations_gt:
            gt.append(
                R.from_euler('ZXY', r, degrees=True).as_quat()[:, [3, 0, 1, 2]])
        rotations_gt_quat = np.stack(gt, axis=0)
        
        if rotations_pred.shape != rotations_gt_quat.shape:
            raise ValueError(
                '`rotations_pred` shape should be %s, got %s' % \
                (rotations_gt_quat.shape, rotations_pred.shape))

        per_pose_error = np.sqrt(((rotations_gt_quat - rotations_pred) ** 2).sum(2)).mean(1)

        subject_scores = {
            'Average': evaluate_by_actions(self, per_pose_error)
        }

        for subject_idx in range(len(self.labels['subject_names'])):
            subject_mask = self.labels['table']['subject_idx'] == subject_idx
            subject_scores[self.labels['subject_names'][subject_idx]] = \
                evaluate_by_actions(self, per_pose_error, subject_mask)

        result = {
            'per_pose_error': subject_scores
        }

        return result['per_pose_error']['Average']['Average'], result