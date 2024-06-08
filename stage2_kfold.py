import numpy as np
import pandas as pd
import heapq
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

#target_list = ['PD',]        #'PD','SP',
def main(target_list):
    for target in target_list:
        best_R2 = [[],[],[]]
        save_R2 = [[],[],[]]
        data = pd.read_excel(f'stage2_excels/{target}/{target}_merge_data.xlsx')
        nor = data.loc[:,'nor'].values
        all_tags = data.columns
        #feature = data.iloc[:, 1:19]
        #print(nor)
        for i in range(4, len(all_tags)):
            feature = data.iloc[:, i].values
            sort = np.argsort(feature)
            X = feature[sort].reshape(-1,1)
            y = nor[sort]
            degree = [1, 2, 3]
            tag = all_tags[i]
            for index, num in enumerate(degree):
                R2 = []
                regressor = make_pipeline(PolynomialFeatures(num), LinearRegression())
                kfold = KFold(n_splits=5, shuffle=True, random_state=4)
                for train_index, test_index in kfold.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    regressor.fit(X_train,y_train)
                    R2.append((regressor.score(X_test, y_test)))
                best_R2[0].append(tag)
                best_R2[1].append(num)
                #best_R2[1].append(round((np.mean(R2)),3))
                best_R2[2].append(np.mean(R2))
                #print(f'\n{tag} degree{num} average-R2:{round((np.mean(R2)),3)}')    
                #print(f'R2:{R2}')
                #print(f'average-R2:{round((np.mean(R2)),3)}')
        #best3 = map(best_R2.index, heapq.nlargest(3, best_R2))
        #print(best_R2)
        print(target)
        best3 = np.array(best_R2[2])
        b = best3.argsort()
        length = len(b)
        for i in range(length-1,0,-1):
            print("%-20s %d %10.3f"%(best_R2[0][b[i]],best_R2[1][b[i]],best_R2[2][b[i]]))
            save_R2[0].append(best_R2[0][b[i]])
            save_R2[1].append(best_R2[1][b[i]])
            save_R2[2].append(best_R2[2][b[i]])
        df = pd.DataFrame({"tag":save_R2[0], "degree":save_R2[1], "R2":save_R2[2]})
        file_path = f'stage2_excels/{target}/{target}_kfold_one.xlsx'     # 輸出excel檔案名稱
        with pd.ExcelWriter(file_path, engine = 'openpyxl', mode = 'w') as writer:
            df.to_excel(writer, sheet_name=target, index = False)
            print(f'{target} kflod saved.')