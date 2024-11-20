#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
import torch
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, getProjectionMatrix, ndc2Pix
import numpy as np
import json
import imageio
import tempfile
import trimesh
import uuid
from glob import glob
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import Intrinsics
from utils.image_utils import load_img
from tqdm import tqdm
from utils.camera_utils_multinerf import generate_interpolated_path

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    is_test: bool
    depth_path: Optional[str] = None
    K: Optional[np.array] = None
    mask: Optional[np.array] = None
    white_background: Optional[bool] = False

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

def readBrics(datadir, split, time: int = 0, downsample: int = 1, white_background: bool = True, opencv_camera=True, load_image_on_the_fly = False):
    # per_cam_poses, intrinsics, cam_ids = load_brics_poses(datadir, downsample=downsample, split=split, opencv_camera=True)
    assert split in ['train', 'test', 'org']

    # load meta data
    with open(os.path.join(datadir, f"transforms_{split}.json"), 'r') as fp:
        meta = json.load(fp)
    frames = meta['frames']
    w, h = int(frames[0]['w']), int(frames[0]['h'])

    # load intrinsics
    intrinsics = Intrinsics(w, h, frames[0]['fl_x'], frames[0]['fl_y'], frames[0]['cx'], frames[0]['cy'], [], [], [], [] )
    for i in range(0, len(frames)):
        intrinsics.append(frames[i]['fl_x'], frames[i]['fl_y'], frames[i]['cx'], frames[i]['cy'])
    intrinsics.scale(1/downsample)

    # load poses
    cam_ids, poses = [], []
    for i in list(range(0, len(frames))):
        pose = np.array(frames[i]['transform_matrix'])
        if opencv_camera: # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            pose[:3, 1:3] *= -1
        poses.append(pose)
        cam_ids.append(frames[i]['file_path'].split('/')[-2])
    per_cam_poses = np.stack(poses)

    # load images and parse cameras
    cam_infos = []
    camera_dict = {}
    uid = 0
    for cam_idx in range(len(cam_ids)):
        cam_name = cam_ids[cam_idx]
        img_path = os.path.join(datadir, "frames_1", cam_name,  f"{time:08d}.png")
        # per_cam_imgs.append(img_path)
        timestamp = time
        image_name = os.path.join(cam_name, f"{time:08d}") #Path(os.path.join(f"{cam_name}_{j:06d}").stem

        # load image and mask
        image, mask = load_img(img_path, downsample = downsample, white_background = white_background)
        
        # prep camera parameters
        # cam_idx = idx
        FovY = focal2fov(intrinsics.focal_ys[cam_idx], intrinsics.height)
        FovX = focal2fov(intrinsics.focal_xs[cam_idx], intrinsics.width)
        w2c = np.linalg.inv(np.array(per_cam_poses[cam_idx]))
        R, T = np.transpose(w2c[:3, :3]), w2c[:3, 3]

        K = np.array([[
            intrinsics.focal_xs[cam_idx], 0, intrinsics.center_xs[cam_idx]],
            [0, intrinsics.focal_ys[cam_idx], intrinsics.center_ys[cam_idx]],
            [0, 0, 1]]
        )
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, K=K,
            image = image if not load_image_on_the_fly else None, mask = mask,
            image_path=img_path, image_name=image_name, is_test = True if split == 'test' else False,
            width=image.size[0], height=image.size[1], white_background = white_background, depth_params = None)
        uid += 1
        camera_dict[cam_name] = cam_info # needed for video camera
        cam_infos.append(cam_info)
    return cam_infos, camera_dict


def readBricsSceneInfo(path, num_pts=200_000, white_background=True, frame_time=0, num_t=1, init='hull', create_video_cams=True, load_image_on_the_fly = False):
    print("Reading Brics Info")
    train_cam_infos, train_camera_dict = readBrics(path, split='train', white_background=white_background, time=frame_time, load_image_on_the_fly = load_image_on_the_fly)
    test_cam_infos, _ = readBrics(path, split='test', white_background=white_background, time=frame_time, load_image_on_the_fly = load_image_on_the_fly)

    # init points
    if init == 'hull':
        first_frame_cameras = train_cam_infos
        aabb = -3.0, 3.0
        grid_resolution = 128
        grid = np.linspace(aabb[0], aabb[1], grid_resolution)
        grid = np.meshgrid(grid, grid, grid)
        grid_loc = np.stack(grid, axis=-1).reshape(-1, 3) # n_pts, 3

        # project grid locations to the image plane
        grid = torch.from_numpy(np.concatenate([grid_loc, np.ones_like(grid_loc[:, :1])], axis=-1)).float() # n_pts, 4
        # grid_mask = np.ones_like(grid_loc[:, 0], dtype=bool)
        grid_counter = np.ones_like(grid_loc[:, 0], dtype=int)
        zfar = 100.0
        znear = 0.01
        trans=np.array([0.0, 0.0, 0.0])
        scale=1.0
        for cam in first_frame_cameras:
            world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, trans, scale)).transpose(0, 1)

            if not load_image_on_the_fly:
                H, W = cam.image.size[1], cam.image.size[0]
            else:
                img, mask = load_img(cam.image_path, white_background = white_background)
                H, W = img.size[1], img.size[0]

            projection_matrix =  getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam.FovX, fovY=cam.FovY, K=cam.K, img_h=H, img_w=W).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            # xyzh = torch.from_numpy(np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)).float()
            cam_xyz = grid @ full_proj_transform # (full_proj_transform @ xyzh.T).T
            uv = cam_xyz[:, :2] / cam_xyz[:, 2:3] # xy coords
            # H, W = cam.image.size[1], cam.image.size[0]
            uv = ndc2Pix(uv, np.array([W, H]))
            uv = np.round(uv.numpy()).astype(int)

            valid_inds = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) 
            # _pix_mask = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if not load_image_on_the_fly:
                cam_mask = np.array(cam.mask) # H,W,1
            else:
                cam_mask = np.array(mask) # H,W,1
            # _pix_mask[_pix_mask] = cam_mask[uv[valid_inds][:, 1], uv[valid_inds][:, 0]].reshape(-1) > 0

            _m = cam_mask[uv[valid_inds][:, 1], uv[valid_inds][:, 0]].reshape(-1) > 0
            # grid_mask[valid_inds] = grid_mask[valid_inds] & _m
            grid_counter[valid_inds] = grid_counter[valid_inds] + _m
            print('grid_counter=', np.mean(grid_counter))

            if True:
                cam_img = np.array(cam.image if not load_image_on_the_fly else img).copy()
                red_uv = uv[valid_inds][_m > 0]
                cam_img[red_uv[:, 1], red_uv[:, 0]] = np.array([255, 0, 0])
                # save cam_img
                imageio.imsave(f'./cam_img.png', cam_img)
                # breakpoint()

        grid_mask = grid_counter > 15 # at least 10 cameras should see the point
        xyz = grid[:, :3].numpy()[grid_mask]
        colors = np.random.random((xyz.shape[0], 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros_like(xyz))
        ply_path = os.path.join(tempfile._get_default_tempdir(), f"{next(tempfile._get_candidate_names())}_{str(uuid.uuid4())}.ply") #os.path.join(path, "points3d.ply")

    else:
        raise NotImplementedError

    # sub sample points if needed
    if xyz.shape[0] > num_pts:
        xyz = xyz[np.random.choice(xyz.shape[0], num_pts, replace=False)]
    colors = np.random.random((xyz.shape[0], 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros_like(xyz))
    storePly(ply_path, xyz, colors)

    # create visualization cameras
    video_cameras = []
    if create_video_cams:
        vis_C2W = []
        vis_cam_order = ['cam01', 'cam04', 'cam09', 'cam15', 'cam23', 'cam28', 'cam32', 'cam34', 'cam35', 'cam36', 'cam37'] + ['cam01', 'cam04']
        cam_id_order = [train_camera_dict[vis_cam_id] for vis_cam_id in vis_cam_order]
        for cam in cam_id_order:
            Rt = np.eye(4)
            Rt[:3, :3] = cam.R
            Rt[:3, 3] = cam.T
            vis_C2W.append(np.linalg.inv(Rt))
        vis_C2W = np.stack(vis_C2W)[:, :3, :4]
        # interpolate between cameras
        visualization_poses = generate_interpolated_path(vis_C2W, 50, spline_degree=3, smoothness=0.0, rot_weight=0.01)
        video_cam_centers = []
        # timesteps = list(range(start_t, start_t+num_t))
        timesteps = list(range(0, num_t))
        timesteps_rev = timesteps + timesteps[::-1]
        for _idx, _pose in enumerate(visualization_poses):
            Rt = np.eye(4)
            Rt[:3, :4] = _pose[:3, :4]
            Rt = np.linalg.inv(Rt)
            R = Rt[:3, :3]
            T = Rt[:3, 3]

            video_cameras.append(CameraInfo(
                    uid=_idx,
                    R=R, T=T,
                    FovY=train_cam_infos[0].FovY, FovX=train_cam_infos[0].FovX,
                    image=None, image_path=None, image_name=f"{_idx:05}", is_test = True,
                    width=train_cam_infos[0].width, height=train_cam_infos[0].height, white_background = white_background, depth_params = None
                    # width=train_cam_infos[0].image.size[0], height=train_cam_infos[0].image.size[1],
            ))
            video_cam_centers.append(_pose[:3, 3])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cameras,
                           nerf_normalization=getNerfppNorm(train_cam_infos),
                           ply_path=ply_path,
                           is_nerf_synthetic = False,
                           )
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Brics": readBricsSceneInfo,
}