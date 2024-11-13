import cv2
import numpy as np
import math
import pickle
import ransac

def cylindrical_projection(img, f):

    """对图像进行柱面投影并返回有效的 x 方向范围"""
    h, w = img.shape[:2]
    # 定义柱面展开图的尺寸
    cylindrical_img = np.zeros_like(img)

    mask = np.zeros((h, w), dtype=np.uint8)

    # 图像中心点
    x_c = w / 2
    y_c = h / 2
    # 投影每个像素
    for y in range(h):
        for x in range(w):
            # 将图像坐标转换到柱面坐标
            # x_shift = x - x_c
            # y_shift = y - y_c
            #
            # # 计算柱面投影的角度和高度
            # theta = math.atan(x_shift / f)
            # # h_cyl = y_shift
            # h_cyl = y_shift / math.sqrt(x_shift ** 2 + f ** 2)
            #
            # # 将角度和高度转换到柱面图像平面坐标
            # x_cyl = int(f * theta + x_c)
            # y_cyl = int(h_cyl + y_c)

            x_cyl = int(f * np.arctan((x - w / 2) / f) + w / 2)
            y_cyl = int(f * (y - h / 2) / np.sqrt((x - w / 2) ** 2 + f ** 2) + h / 2)
            # 检查坐标是否在图像范围内
            if 0 <= x_cyl < w and 0 <= y_cyl < h:
                cylindrical_img[y_cyl, x_cyl] = img[y, x]
                mask[y_cyl, x_cyl] = 1  # 标记有效区域

    # 获得左右x_min x_max

    x_indices = np.where(mask.any(axis=0))[0]
    x_min, x_max = x_indices.min(), x_indices.max()

    cylindrical_img = cylindrical_img[:,x_min:x_max+1,:]
    return cylindrical_img, (x_min, x_max)

def get_valid_width(img):
    h, w = img.shape[:2]
    # 图像中心点
    x_c = w / 2
    valid_area = []
    for x in [0,w-1]:
        x_shift = x - x_c
        theta = math.atan(x_shift / f)
        x_cyl = int(f * theta + x_c)
        print(x_cyl)
        valid_area.append(x_cyl)
    return valid_area

def compute_images_offsets(imgs, f):
    """返回移动的距离组"""

    offset = []
    warped_imgs = []
    offset.append([0,0])#第一张图不移动
    length = len(imgs)
    h, w = imgs[0].shape[:2]


    img1 = imgs[0]
    img1_proj, (x1_min, x1_max) = cylindrical_projection(img1, f)
    valid_w,valid_h = img1_proj.shape[:2]
    warped_imgs.append(img1_proj)
    for i in range(length):
        print(i)
        if(i == length - 1):
            img2 = imgs[0]
        else:
            img2 = imgs[i+1]
        img2_proj, (x2_min, x2_max) = cylindrical_projection(img2, f)
        if(i != length - 1):
            warped_imgs.append(img2_proj)
        # 使用SIFT提取特征并进行匹配
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1_proj, None)
        kp2, des2 = sift.detectAndCompute(img2_proj, None)
        # 匹配特征点
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # 使用比率测试筛选匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        # 提取匹配的关键点坐标
        if len(good_matches) > 4:
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pickle.dump(dst_pts, open('dst_pts.pkl', 'wb'))
            pickle.dump(src_pts, open('src_pts.pkl', 'wb'))
            # [5219 1 2]
            # 使用RANSAC找到平移变换
            # M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
            M = ransac.ransac_transfrom(src_pts, dst_pts, 1000, 50)
            # 平移量
            dx, dy = M[0, 2], M[1, 2]
            print("transform matrix(only consider shifting operation): ")
            print(M)
            if(i == length - 1):
                tmp = [-dx + offset[-1][0]+valid_w,-dy+offset[-1][1]]
            else:
                tmp = [dx + offset[-1][0],dy+offset[-1][1]]
            offset.append(tmp)
        else:
            offset.append([0,0])

        img1 = img2
        img1_proj = img2_proj
    return warped_imgs,offset
def stitch_images_o(warp_imgs,offsets):
    valid_h, valid_w = warp_imgs[0].shape[:2]
    panorama_w = int(offsets[-1][0])
    panorama_h = valid_h + int(offsets[-1][1])
    panorama = np.zeros((panorama_h, panorama_w, 3), dtype=warp_imgs[0].dtype)
    length = len(warp_imgs)

    for i in range(length):
        print(i)
        valid_h,valid_w = warp_imgs[i].shape[:2]
        if(i==0):
            panorama[0:valid_h,0:valid_w,:] = warp_imgs[i]
        else:
            x_pre = int(offsets[i-1][0]) + valid_w

            x_start = int(offsets[i][0])
            y_start = int(offsets[i][1])
            # y_start = int(0)

            x_end = x_start + valid_w
            y_end = y_start + valid_h
            guodu = x_pre - x_start
            for x in range(x_start,x_end):
                for y in range(y_start,y_end):
                    if(x<0 or  x>=panorama_w-1 or  y<0 or  y>=panorama_h-1):
                        continue
                    if(warp_imgs[i][y-y_start][x-x_start]==[0,0,0]).all():
                        continue
                    if (panorama[y,x] == [0,0,0]).all():
                        panorama[y,x] = warp_imgs[i][y-y_start][x-x_start]
                    elif x <= x_pre:
                        alp = (x - x_start) / guodu
                        panorama[y,x] = (1-alp)*panorama[y,x] + alp * warp_imgs[i][y-y_start][x-x_start]
                    else:
                        panorama[y,x] = warp_imgs[i][y-y_start][x-x_start]

        cv2.namedWindow("show", cv2.WINDOW_NORMAL);
        cv2.imshow("show",panorama)
        # 等待用户按键，参数为等待时间（毫秒）
        cv2.waitKey(0)
        # 销毁所有 OpenCV 创建的窗口
        cv2.destroyAllWindows()
    return panorama






def cylindrical_to_normal(cylindrical_img, f):
    """将柱面投影图像恢复到平面坐标"""
    h, w = cylindrical_img.shape[:2]
    normal_img = np.zeros_like(cylindrical_img)
    x_c = w / 2
    y_c = h / 2

    for y_cyl in range(h):
        for x_cyl in range(w):
            # 逆向柱面投影变换
            theta = (x_cyl - x_c) / f
            h_plane = y_cyl - y_c

            # 计算原始平面坐标
            x = int(f * math.tan(theta) + x_c)
            y = int(h_plane + y_c)

            # 检查坐标是否在正常图像范围内
            if 0 <= x < w and 0 <= y < h:
                normal_img[y, x] = cylindrical_img[y_cyl, x_cyl]

    return normal_img


def focus_lenth_correct(f,l):

    print("l:", l)
    c = 2 * math.pi * f
    gap = c - l
    print("c:", c)
    theta_g = gap / f
    f_p = f * (1 - (theta_g / (2 * math.pi)))
    print("suggest f:", f_p)


if __name__ == "__main__":
    command = "use pre-data"
    # command = "run data"

    f = 2230  # 设置适当的焦距
    # 加载图像
    # w = 1536
    images = [f'pano1/100NIKON-DSCN00{i:02d}_DSCN00{i:02d}.JPG' for i in range(25, 8,-1)]
    image_list = [cv2.imread(image_path) for image_path in images]

    valid_x_area = get_valid_width(image_list[0])
    h, w = image_list[0].shape[:2]
    print(valid_x_area)
    valid_length = valid_x_area[1] - valid_x_area[0] + 1

    if command == "use pre_data":
        # 使用 pickle 反序列化数据
        with open('warped_imgs.pkl', 'rb') as file:
            warped_imgs = pickle.load(file)
        with open('offset.pkl', 'rb') as file:
            offset = pickle.load(file)
    else:
        # 得到变换的图片组和对应的偏移
        warped_imgs, offset = compute_images_offsets(image_list, f)




    print("picture offset : ", offset)
    focus_lenth_correct(f,offset[-1][0])

    # 保存中间结果
    pickle.dump(warped_imgs, open('warped_imgs.pkl', 'wb'))
    pickle.dump(offset,open('offset.pkl', 'wb'))

    panorama = stitch_images_o(warped_imgs,offset)
    cv2.imwrite("panorama_myrascas",panorama)