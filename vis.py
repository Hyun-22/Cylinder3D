import open3d as o3d
from glob import glob
import os

if __name__ == "__main__":
    # result_paths_1 = glob("/home/ailab/Desktop/git/Cylinder3D/logs/train_normal_valid_normal/result/*.pcd")
    # result_paths_2 = glob("/home/ailab/Desktop/git/Cylinder3D/logs/train_normal_valid_ours_snow/result/*.pcd")
    # result_paths_3 = glob("/home/ailab/Desktop/git/Cylinder3D/logs/train_ours_valid_ours_snow/result/*.pcd")
    
    result_paths_1 = glob("/home/ailab/Desktop/git/Cylinder3D/logs/train_normal_valid_normal/result_urban/*.pcd")
    result_paths_2 = glob("/home/ailab/Desktop/git/Cylinder3D/logs/train_ours_valid_ours_snow/result_urban/*.pcd")
    result_paths_1.sort()
    result_paths_2.sort()

    # result_paths_1 = result_paths_1[-40:]
    # result_paths_2 = result_paths_2[-40:]
    # result_paths_3.sort()
    
    # for result_path_1, result_path_2, result_path_3 in zip(result_paths_1, result_paths_2, result_paths_3):
    #     result_1 = o3d.io.read_point_cloud(result_path_1)
    #     result_2 = o3d.io.read_point_cloud(result_path_2)
    #     result_3 = o3d.io.read_point_cloud(result_path_3)
        
    #     o3d.visualization.draw_geometries([result_1])
    #     o3d.visualization.draw_geometries([result_2])
    #     o3d.visualization.draw_geometries([result_3])
    #     print("================================")
    for result_path_1, result_path_2 in zip(result_paths_1, result_paths_2):
        result_1 = o3d.io.read_point_cloud(result_path_1)
        result_2 = o3d.io.read_point_cloud(result_path_2)
    
        
        o3d.visualization.draw_geometries([result_1])
        o3d.visualization.draw_geometries([result_2])
        print("================================")
    