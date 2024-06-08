import numpy as np
import pandas as pd
import heapq
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

#target_list = ['PD']        #'PD','SP',
def main(target_list):
    for target in target_list:
        best_R2 = [[],[],[],[]]
        save_R2 = [[],[],[],[]]
        data = pd.read_excel(f'stage2_excels/{target}/{target}_merge_data.xlsx')
        y = data['nor']
        all_tags = data.columns
        for i in range(4, len(all_tags)):
            if i == len(all_tags) - 1:
                break
            for j in range(len(all_tags)-i):
                tag1 = all_tags[i]
                tag2 = all_tags[i+j]
                X = np.column_stack((data[tag1],data[tag2]))
                degree = [1, 2, 3]
                for index, num in enumerate(degree):
                    R2 = []
                    regressor = make_pipeline(PolynomialFeatures(num), LinearRegression())
                    kfold = KFold(n_splits=5, shuffle=True, random_state=4)
                    for train_index, test_index in kfold.split(X):
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        regressor.fit(X_train,y_train)
                        R2.append((regressor.score(X_test, y_test)))
                    if np.mean(R2) > 0:
                        best_R2[0].append(tag1)
                        best_R2[1].append(tag2)
                        best_R2[2].append(num)
                        best_R2[3].append(np.mean(R2))
        print(target)
        best3 = np.array(best_R2[3])
        b = best3.argsort()
        length = len(b)
        if length >= 5:
            start = length -1
            stop = length -6
        else :
            start = length - 1
            stop = -1
        for i in range(start,stop,-1):
            print("%-20s %-20s %d %10.3f"%(best_R2[0][b[i]],best_R2[1][b[i]],best_R2[2][b[i]],best_R2[3][b[i]]))
            save_R2[0].append(best_R2[0][b[i]])
            save_R2[1].append(best_R2[1][b[i]])
            save_R2[2].append(best_R2[2][b[i]])
            save_R2[3].append(best_R2[3][b[i]])
        df = pd.DataFrame({"tag1":save_R2[0],"tag2":save_R2[1], "degree":save_R2[2], "R2":save_R2[3]})
        file_path = f'stage2_excels/{target}/{target}_kfold_two.xlsx'     # 輸出excel檔案名稱
        with pd.ExcelWriter(file_path, engine = 'openpyxl', mode = 'w') as writer:
            df.to_excel(writer, sheet_name=target, index = False)
            print(f'{target} kflod saved.')