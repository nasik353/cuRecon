import argparse

import torch
import numpy as np

from pareconv.utils.data import registration_collate_fn_stack_mode, precompute_neibors
from pareconv.utils.torch import to_cuda, release_cuda
from pareconv.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from pareconv.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model

def read_from_ply(filename):
    points = []
    colors = []
    
    with open(filename, 'r') as file:
        num_vertices = 0
        while True:
            line = file.readline().strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line == 'end_header':
                break
        
        for _ in range(num_vertices):
            line = file.readline().strip()
            values = line.split()
            
            points.append([float(x) for x in values[:3]])
            
            colors.append([int(x)/255.0 for x in values[3:6]])
    
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    
    return points, colors


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", default='/home/jt/projects/cuRecon/build/frame_1.ply', help="src point cloud numpy file")
    parser.add_argument("--ref_file", default='/home/jt/projects/cuRecon/build/frame_0.ply', help="src point cloud numpy file")
    parser.add_argument("--gt_file", default='/home/jt/.src/PARENet/data/demo/gt.npy', help="ground-truth transformation file")
    parser.add_argument("--weights", default='/home/jt/.src/PARENet/pretrain/3dmatch.pth.tar', help="model weights file")
    return parser


def load_data(args):
    src_points, src_colors = read_from_ply(args.src_file)
    ref_points, ref_colors = read_from_ply(args.ref_file)
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_colors": ref_colors.astype(np.float32),
        "src_colors": src_colors.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    # if args.gt_file is not None:
    #     transform = np.load(args.gt_file)
    #     data_dict["transform"] = transform.astype(np.float32)

    return data_dict


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()

    # prepare data
    data_dict = load_data(args)
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.num_neighbors, cfg.backbone.subsample_ratio
    )

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])

    # prediction
    data_dict = to_cuda(data_dict)
    data = precompute_neibors(data_dict['points'], data_dict['lengths'],
                              cfg.backbone.num_stages,
                              cfg.backbone.num_neighbors,
                              )

    data_dict.update(data)
    output_dict = model(data_dict)

    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    ref_points = output_dict["ref_points"]
    ref_colors = output_dict["ref_colors"]
    src_points = output_dict["src_points"]
    src_colors = output_dict["src_colors"]
    estimated_transform = output_dict["estimated_transform"]


    # visualization
    print("visualization0")
    print(ref_points.shape)
    ref_pcd = make_open3d_point_cloud(ref_points, ref_colors)
    ref_pcd.estimate_normals()
    # ref_pcd.paint_uniform_color(get_color("custom_yellow"))
    src_pcd = make_open3d_point_cloud(src_points, src_colors)
    src_pcd.estimate_normals()
    # src_pcd.paint_uniform_color(get_color("custom_blue"))
    draw_geometries(ref_pcd, src_pcd)
    src_pcd = src_pcd.transform(estimated_transform)
    draw_geometries(ref_pcd, src_pcd)

    # # compute error
    # if args.gt_file is not None:
    #     transform = data_dict["transform"]
    #     rre, rte = compute_registration_error(transform, estimated_transform)
    #     print(f"RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}")


if __name__ == "__main__":
    main()
