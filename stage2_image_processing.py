import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import copy

# 圖像經過閥值處理，'計算面積'和顏色通道的各'種統計指標'，原始圖像以及數據的excel
# 輸出color_space、ROI圖片
def main(target_list):
    
    def find_ROI(picture):      # 讀取相片選取出ROI 輸入相片路徑 回傳ROI 原始圖像
        image = cv2.imread(picture)     # 讀取資料源圖像
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)      # 將圖像轉換為HSV色彩空間
        lower_threshold = np.array([0, 0, 0])     # 定義HSV低範圍
        if target == 'GA':
            upper_threshold = np.array([27, 255, 255])      # 定義HSV高範圍
        else :
            upper_threshold = np.array([180, 255, 255])      # 定義HSV高範圍
        mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)     # 創建一個遮罩，保留HSV範圍內的部分
        result = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)       # 在原始圖像上使用遮罩留下ROI
        #result = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return  result ,image     # 回傳ROI、原始圖片

    def save_the_ROI(data_path4, result):       # 儲存ROI影像 輸入儲存路徑 ROI影像
        #filename, extension = os.path.splitext(data_path4)       # 取出儲存位址 移除副檔名
        #new_picture = f"{filename}.jpg"     # 設定檔案名稱及位址
        output = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)        # 影像轉回BGR色彩空間
        cv2.imwrite(data_path4, output)        # 儲存影像
        
    def count_area(mask2, target):       # 計算原始圖像面積 輸入原始圖像 回傳面積
        
        area = 0        # 宣告計算面積為0
        mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)     # 將原始影像轉為灰值         
        height, width = mask2.shape     # 2mm 231像素點
        for i in range(height):     # 長
            for j in range(width):      # 寬
                if mask2[i, j] != 0:        # 像素不為0
                    area += 1       # 計算一個像素點
        if target == 'SP':
            area = area / ((22.27)**2)
        else:    
            area = area / ((115.5)**2)      # 將面積從像素量轉為 mm平方
        
        return area     # 回傳面積數值

    def Statistical(channel):       # 讀取單一個通道的所有數值 並計算所有統計指標 輸入通道值 回傳所有統計指標
        copied_list = copy.deepcopy(channel)        # deepcopy傳入的通道 以免影響原本的資料
        arr = copied_list[copied_list != 0]     # 去除為0的像素
        mean = np.mean(arr)     # 計算通道所有數值的平均值
        median = np.median(arr)     # 計算通道所有數值的中位數
        variance = np.var(arr)      # 計算通道所有數值的變異數
        std_dev = np.std(arr)       # 計算通道所有數值的標準差
        percentile_25 = np.percentile(arr, 25)      # 計算通道所有數值的25百分位數
        percentile_75 = np.percentile(arr, 75)      # 計算通道所有數值的75百分位數
        all = [mean, median, variance, std_dev, percentile_25, percentile_75]
        return all      # 回傳所有統計指標

    def statistical_indicator_processing(result, i):      # 轉換顏色空間 儲存所有通道的統計指標到list 輸入ROI圖片
        if i != 0:
            result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)        # 轉回BGR
        if i == 1:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)        # 轉為其他顏色空間！！！！！
        if i == 2:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2XYZ)        # 轉為其他顏色空間！！！！！
        # 讀取個通道值
        channel1 = result[:, :, 0]       # 第一個通道
        channel2 = result[:, :, 1]        # 第二個通道
        channel3 = result[:, :, 2]     # 第三個通道
        # 計算通道各項統計指標
        Statistical1 = Statistical(channel1)        # 計算第一個通道的統計指標
        Statistical2 = Statistical(channel2)     # 計算第二個通道的統計指標
        Statistical3 = Statistical(channel3)      # 計算第三個通道的統計指標
        #將各項統計指標儲存到個別的list
        if i == 0:
            hsv_h.append(Statistical1)        # 第一個通道平均值加入lsit
            hsv_s.append(Statistical2)        # 第二個通道平均值加入lsit
            hsv_v.append(Statistical3)        # 第三個通道平均值加入lsit
        if i == 1:
            yuv_y.append(Statistical1)        # 第一個通道平均值加入lsit
            yuv_u.append(Statistical2)        # 第二個通道平均值加入lsit
            yuv_v.append(Statistical3)        # 第三個通道平均值加入lsit
        if i == 2:
            xyz_x.append(Statistical1)        # 第一個通道平均值加入lsit
            xyz_y.append(Statistical2)        # 第二個通道平均值加入lsit
            xyz_z.append(Statistical3)        # 第三個通道平均值加入lsit


    def excel_data(picture, picture_area):      # 照片名稱、面積 加入list 輸入圖片名稱 原始圖片面積
        picture_name = os.path.splitext(picture)      # 取出照片名稱（不含位址）
        name.append(picture_name[0])        # 將名稱加入 name list
        area.append(picture_area)       # 將面積加入 area list

    def save_excel_data(name,area,
                        hsv_h,hsv_s,hsv_v,yuv_y,yuv_u,yuv_v,xyz_x,xyz_y,xyz_z):       
        # 儲存所有值到excel檔案當中的個別種類工作表中
        # excel column 名稱
        hsv_h = np.array(hsv_h)       # mean,median,variance,std_dev,percentile_25,percentile_75      # mean,median,variance,std_dev,percentile_25,percentile_75
        hsv_s = np.array(hsv_s) 
        hsv_v = np.array(hsv_v) 
        yuv_y = np.array(yuv_y) 
        yuv_u = np.array(yuv_u) 
        yuv_v = np.array(yuv_v)  
        xyz_x = np.array(xyz_x) 
        xyz_y = np.array(xyz_y) 
        xyz_z = np.array(xyz_z) 
         
        file_path = f"stage2_excels/{target}/{target}_color_space.xlsx"        # excel 儲存位址
        column = ["name","area",
                "hsv_h_mean","hsv_h_median","hsv_h_variance","hsv_h_std_dev","hsv_h_25","hsv_h_75",
                "hsv_s_mean","hsv_s_median","hsv_s_variance","hsv_s_std_dev","hsv_s_25","hsv_s_75",
                "hsv_v_mean","hsv_v_median","hsv_v_variance","hsv_v_std_dev","hsv_v_25","hsv_v_75",
                "yuv_y_mean","yuv_y_median","yuv_y_variance","yuv_y_std_dev","yuv_y_25","yuv_y_75",
                "yuv_u_mean","yuv_u_median","yuv_u_variance","yuv_u_std_dev","yuv_u_25","yuv_u_75",
                "yuv_v_mean","yuv_v_median","yuv_v_variance","yuv_v_std_dev","yuv_v_25","yuv_v_75",
                "xyz_x_mean","xyz_x_median","xyz_x_variance","xyz_x_std_dev","xyz_x_25","xyz_x_75",
                "xyz_y_mean","xyz_y_median","xyz_y_variance","xyz_y_std_dev","xyz_y_25","xyz_y_75",
                "xyz_z_mean","xyz_z_median","xyz_z_variance","xyz_z_std_dev","xyz_z_25","xyz_z_75",]
        data_value = np.column_stack((name, area,
                    hsv_h[:,0], hsv_h[:,1], hsv_h[:,2], hsv_h[:,3], hsv_h[:,4], hsv_h[:,5],
                    hsv_s[:,0], hsv_s[:,1], hsv_s[:,2], hsv_s[:,3], hsv_s[:,4], hsv_s[:,5],
                    hsv_v[:,0], hsv_v[:,1], hsv_v[:,2], hsv_v[:,3], hsv_v[:,4], hsv_v[:,5],
                    yuv_y[:,0], yuv_y[:,1], yuv_y[:,2], yuv_y[:,3], yuv_y[:,4], yuv_y[:,5],
                    yuv_u[:,0], yuv_u[:,1], yuv_u[:,2], yuv_u[:,3], yuv_u[:,4], yuv_u[:,5],
                    yuv_v[:,0], yuv_v[:,1], yuv_v[:,2], yuv_v[:,3], yuv_v[:,4], yuv_v[:,5],
                    xyz_x[:,0], xyz_x[:,1], xyz_x[:,2], xyz_x[:,3], xyz_x[:,4], xyz_x[:,5],
                    xyz_y[:,0], xyz_y[:,1], xyz_y[:,2], xyz_y[:,3], xyz_y[:,4], xyz_y[:,5],
                    xyz_z[:,0], xyz_z[:,1], xyz_z[:,2], xyz_z[:,3], xyz_z[:,4], xyz_z[:,5],))
        data = pd.DataFrame(data_value ,columns=column)       # 所有資料變成dataframe
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:     # 設定？
            data.to_excel(writer, sheet_name=target, index=False)       # 將資料寫入excel
            print("Excel file saved.")

    # main
    # 宣告各項list    
    #target_list = ['PD','SP','GA'] #'PD','SP',
    for target in target_list:
        name = []       
        area = []
        hsv_h = []      # mean,median,variance,std_dev,percentile_25,percentile_75
        hsv_s = []
        hsv_v = []
        yuv_y = []
        yuv_u = []
        yuv_v = []
        xyz_x = []
        xyz_y = []
        xyz_z = []

        count = 0     # 計數幾張照片
        # 讀取資料夾內的照片
        source_path1 = f'stage2_data/{target}'       # 讀取資料源第一個資料夾
        dirs1 = os.listdir( source_path1 )      # image下有哪些資料夾 
        data_path1 = f'stage2_ROI/{target}'       # 儲存資料夾
        if not os.path.exists(data_path1):      # 製作儲存data資料夾
            os.makedirs(data_path1)    
        for picture in tqdm(dirs1, desc='pictures', unit='items' ):     # 處理 第二層下的照片（照片）
            if picture == '.DS_Store' : continue        # macos資料夾錯誤
            path2 = os.path.join(source_path1, picture)     # 結合路徑 (資料源資料夾＋物種資料夾)＋照片
            data_path2 = os.path.join(data_path1, picture)      # 結合路徑 (儲存data資料夾＋物種資料夾)+照片
            count += 1      # 計算照片數量
            #圖片處理
            result , image = find_ROI(path2)       # 讀取相片選取出ROI 輸入相片路徑 回傳ROI圖片 原始圖像
            save_the_ROI(data_path2, result)        # 儲存ROI影像 輸入儲存路徑 ROI圖片 # 可以關閉
            picture_area = count_area(image, target)        # 計算原始圖像面積 輸入原始圖像 回傳面積
            for i in range(3):
                statistical_indicator_processing(result, i)       # 轉換顏色空間 儲存所有通道的統計指標到list 輸入ROI圖片
            excel_data(picture, picture_area)       # 照片名稱、面積 加入list 輸入圖片名稱 原始圖片面積
        time.sleep(0.1)  # 模拟操作耗时
        # 儲存所有值到excel檔案當中的個別種類工作表中
        save_excel_data(name,area,
                        hsv_h,hsv_s,hsv_v,yuv_y,yuv_u,yuv_v,xyz_x,xyz_y,xyz_z)         
        print(count)    #   图片個数
