from copy import deepcopy
import numpy as np

import torch
from mvn.models.v2v import Basic3DBlock, Res3DBlock, Pool3DBlock
import torch.nn as nn
import torch.nn.functional as F
from mvn.utils import op, volumetric


from mvn.models import pose_resnet

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res3 = Res3DBlock(128, 128)
        self.encoder_pool4 = Pool3DBlock(2)
        self.encoder_res4 = Res3DBlock(128, 128)
        self.encoder_pool5 = Pool3DBlock(2)
        self.encoder_res5 = Res3DBlock(128, 128)

    def forward(self, x):
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)

        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)

        x = self.encoder_pool3(x)
        x = self.encoder_res3(x)
        
        x = self.encoder_pool4(x)
        x = self.encoder_res4(x)
        
        x = self.encoder_pool5(x)
        x = self.encoder_res5(x)

        return x

class Regressor(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, out_features)

        self.bn = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.bn(self.fc1(x))
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

# class Regressor(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()

#         self.linear = nn.Linear(in_features, 1024)
#         self.relu = nn.ReLU(inplace=True)
#         self.residual_linear1 = ResidualLinear(1024)
#         self.residual_linear2 = ResidualLinear(1024)
#         self.out = nn.Linear(1024, out_features)

#     def forward(self, x):
#         x = self.relu(self.linear(x))
#         x = self.residual_linear1(x)
#         x = self.residual_linear2(x)
#         x = self.out(x)
#         return x

# class ResidualLinear(nn.Module):
#     def __init__(self, features):
#         super().__init__()

#         self.layers = nn.Sequential(
#             nn.Linear(features, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, features),
#             nn.BatchNorm1d(features),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         z = self.layers(x)
#         out = z + x
#         return out

class V2VRegressor(nn.Module):
    def __init__(self, input_channels, output_features):
        super().__init__()

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 16, 7),
            Res3DBlock(16, 32),
            Res3DBlock(32, 32),
            Res3DBlock(32, 32)
        )

        self.encoder = Encoder()

        self.back_layers = nn.Sequential(
            Res3DBlock(128, 64),
            Basic3DBlock(64, 32, 1),
            Basic3DBlock(32, 32, 1),
        )

        self.regressor = Regressor(32 * 2 * 2 * 2, output_features)

        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder(x)
        x = self.back_layers(x)
        # flatten
        x = torch.flatten(x, 1)
        x = self.regressor(x)        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)

class VolumetricAngleRegressor(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_aggregation_method

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size

        self.cuboid_side = config.model.cuboid_side

        self.kind = config.model.kind
        self.use_gt_pelvis = config.model.use_gt_pelvis

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        # modules
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)

        for p in self.backbone.final_layer.parameters():
            p.requires_grad = False

        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        ) # 1x1 Conv2d to squeeze the channel size before procesing volumes

        self.volume_net = V2VRegressor(32, self.num_joints * 4) # quaternions


    def forward(self, images, proj_matricies, skeleton, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        heatmaps, features, _, vol_confidences = self.backbone(images)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        n_joints = heatmaps.shape[2]

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, heatmap_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        proj_matricies = proj_matricies.float().to(device)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device) # store pelvis coordinates
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device) # b x 64 x 64 x 64 x 3
        for batch_i in range(batch_size):
            # if self.use_precalculated_pelvis:
            if self.use_gt_pelvis:
                keypoints_3d = batch['keypoints_3d'][batch_i]
            else:
                keypoints_3d = batch['pred_keypoints_3d'][batch_i]

            base_point = keypoints_3d[0, :3] # root joint to help create volume

            base_points[batch_i] = torch.from_numpy(base_point).to(device)

            # build cuboid
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side]) # 2500 x 2500 x 2500
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation on axis
            # if self.training:
            #     theta = np.random.uniform(0.0, 2 * np.pi)
            # else:
            #     theta = 0.0

            # if self.kind == "coco":
            #     axis = [0, 1, 0]  # y axis
            # elif self.kind == "mpii":
            #     axis = [0, 0, 1]  # z axis
            theta = 0.0
            axis = [0, 0, 1] # try with non-rotate first

            center = torch.from_numpy(base_point).type(torch.float).to(device)

            # rotate
            coord_volume = coord_volume - center # translate to local coordinates
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis) # random rotation
            coord_volume = coord_volume + center # translate back to global coordinates

            # transfer
            if self.transfer_cmu_to_human36m:  # different world coordinates
                coord_volume = coord_volume.permute(0, 2, 1, 3)
                inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
                coord_volume = coord_volume.index_select(1, inv_idx)

            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])

        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

        # integral 3d
        rotations = self.volume_net(volumes) # output quaternions
        rotations = rotations.view(-1, self.num_joints, 4)
        normalized = F.normalize(rotations, dim=2).view(rotations.shape) # normalize quaternions to get unit quaternions

        keypoints_pred = skeleton.forward_kinematics(normalized, base_points)

        return keypoints_pred, normalized, heatmaps

if __name__ == '__main__':
    regressor = V2VRegressor(64, 16)
    X = torch.randn(2, 64, 64, 64, 64)
    y = regressor(X)
    print(y.size())