import pandas as pd
import numpy as np

# 共生藻數量excel與找片顏色空間數據excel合併成一個新的excel
# 讀取color_space、symbiodinium
# 輸出marge_data
#target_list = ['PD',] #'PD','SP',
def main(target_list):
    for target in target_list:
        symbiodinium_excel = pd.read_excel("stage2_excels/all_symbiodinium.xlsx", engine='openpyxl')        # 讀取共生藻excel檔案
        x1 = symbiodinium_excel.iloc[1:,:].values.tolist()      # 讀取共生藻檔案除了column名稱以外的資料
        #x2 = symbiodinium_excel.iloc[1:,-1:].values.tolist()       
        #list_x_all = []

        color_space_excel = pd.read_excel(f"stage2_excels/{target}/{target}_color_space.xlsx")      # 讀取color space的工作表
        y = color_space_excel.iloc[:,:].values       # 讀取所有column與roll的資料
        list_y = y.tolist()     # 將資料轉成list型態    
        list_excel = []     # 建立一個空list 後面要加入新的excel檔
        # 搜索共生藻與色彩空間兩檔案中相同的相片
        for i in x1:        # 瀏覽共生藻檔案
            for j in list_y:        # 瀏覽color space檔案
                if i[0] == j[0]:        # 比對相同的名稱
                    #print(i,j)
                    if i[8] == 0:       # 共生藻數量平均為0
                        nor = 0         # 將nor設為0
                    else:       # 共生藻數量平均不為0    
                        nor = i[8] / j[1]       # 共生藻數量平均/area
                    list_excel.append([j[0],i[8],j[1],nor,
                                        j[2], j[3], j[4], j[5], j[6], j[7],
                                        j[8], j[9], j[10], j[11], j[12], j[13],
                                        j[14], j[15], j[16], j[17], j[18], j[19],
                                        j[20], j[21], j[22], j[23], j[24], j[25],
                                        j[26], j[27], j[28], j[29], j[30], j[31],
                                        j[32], j[33], j[34], j[35], j[36], j[37],
                                        j[38], j[39], j[40], j[41], j[42], j[43],
                                        j[44], j[45], j[46], j[47], j[48], j[49],
                                        j[50], j[51], j[52], j[53], j[54], j[55],
                                        j[56], j[57], j[58], j[59], j[60], j[61],
                                        j[62], j[63], j[64], j[65], j[66], j[67],
                                        j[68], j[69], j[70], j[71], j[72], j[73],
                                        j[74], j[75], j[76], j[77], j[78], j[79],
                                        j[80], j[81], j[82], j[83], j[84], j[85],
                                        j[86], j[87], j[88], j[89], j[90], j[91]])



        df = pd.DataFrame(list_excel, columns=["name","symbiodinium", "area", "nor",
                                            "hsv_h_mean","hsv_h_median","hsv_h_variance","hsv_h_std_dev","hsv_h_percentile_25","hsv_h_percentile_75",
                                                "hsv_s_mean","hsv_s_median","hsv_s_variance","hsv_s_std_dev","hsv_s_percentile_25","hsv_s_percentile_75",
                                                "hsv_v_mean","hsv_v_median","hsv_v_variance","hsv_v_std_dev","hsv_v_percentile_25","hsv_v_percentile_75",
                                                "lab_l_mean","lab_l_median","lab_l_variance","lab_l_std_dev","lab_l_percentile_25","lab_l_percentile_75",
                                                "lab_a_mean","lab_a_median","lab_a_variance","lab_a_std_dev","lab_a_percentile_25","lab_a_percentile_75",
                                                "lab_b_mean","lab_b_median","lab_b_variance","lab_b_std_dev","lab_b_percentile_25","lab_b_percentile_75",
                                                "luv_l_mean","luv_l_median","luv_l_variance","luv_l_std_dev","luv_l_percentile_25","luv_l_percentile_75",
                                                "luv_u_mean","luv_u_median","luv_u_variance","luv_u_std_dev","luv_u_percentile_25","luv_u_percentile_75",
                                                "luv_v_mean","luv_v_median","luv_v_variance","luv_v_std_dev","luv_v_percentile_25","luv_v_percentile_75",
                                                "xyz_x_mean","xyz_x_median","xyz_x_variance","xyz_x_std_dev","xyz_x_percentile_25","xyz_x_percentile_75",
                                                "xyz_y_mean","xyz_y_median","xyz_y_variance","xyz_y_std_dev","xyz_y_percentile_25","xyz_y_percentile_75",
                                                "xyz_z_mean","xyz_z_median","xyz_z_variance","xyz_z_std_dev","xyz_z_percentile_25","xyz_z_percentile_75",
                                                "yuv_y_mean","yuv_y_median","yuv_y_variance","yuv_y_std_dev","yuv_y_percentile_25","yuv_y_percentile_75",
                                                "yuv_u_mean","yuv_u_median","yuv_u_variance","yuv_u_std_dev","yuv_u_percentile_25","yuv_u_percentile_75",
                                                "yuv_v_mean","yuv_v_median","yuv_v_variance","yuv_v_std_dev","yuv_v_percentile_25","yuv_v_percentile_75"])      # 設定資料為dataframe column名稱設定
        excel_file = f'stage2_excels/{target}/{target}_merge_data.xlsx'     # 輸出excel檔案名稱

        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='w') as writer:     # 設定？
            df.to_excel(writer, index=False)       # 將資料寫入excel
            print(f"{target} Excel file saved.") 