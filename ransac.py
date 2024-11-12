import math
import random
import numpy as np
import pickle

def ransac_transfrom(src_pts, dst_pts, max_iter = 1000, inlier_threshold = 50):
    """RASCAS算法，输入为配对好的源点集合和目标点几何，迭代次数，阈值，前两项参数匹配了opencv的形式"""
    best_tx,best_ty = 0,0
    max_inlier = 0
    for iter in range(max_iter):

        progress = (iter/max_iter) * 100

        # 每隔10%打印进度
        if progress % 10 == 0:
            print(f"Rascas Progress: {progress:.2f}%")
            # print("Rascas Progress: ",progress)
        rand_idx = random.randint(0,len(src_pts)-1)
        src_pt = src_pts[rand_idx][0]
        dst_pt = dst_pts[rand_idx][0]

        tx = dst_pt[0] - src_pt[0]
        ty = dst_pt[1] - src_pt[1]

        inlier_cnt = 0

        for i_idx in range(len(src_pts)):
            src_test_pt = src_pts[i_idx][0]
            dst_test_pt = dst_pts[i_idx][0]
            cx = dst_test_pt[0] - src_test_pt[0]
            cy = dst_test_pt[1] - src_test_pt[1]

            if np.sqrt((cx - tx)**2 + (cy - ty)**2) < inlier_threshold:
                inlier_cnt += 1

            if(inlier_cnt>max_inlier):
                max_inlier = inlier_cnt
                best_tx,best_ty = tx,ty
                # print(best_tx,best_ty,max_inlier)

    M = np.zeros((3,3))
    M[0,0] = 1
    M[1,1] = 1
    M[2,2] = 1

    M[0,2] = best_tx
    M[1,2] = best_ty

    return M

def use_rascas():
    # 使用 pickle 反序列化数据
    with open('dst_pts.pkl', 'rb') as file:
        dst_pts = pickle.load(file)
    with open('src_pts.pkl', 'rb') as file:
        src_pts = pickle.load(file)
    M = ransac_transfrom(src_pts, dst_pts)
    print(M)

