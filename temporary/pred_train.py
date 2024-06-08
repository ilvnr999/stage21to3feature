import numpy as np
import pandas as pd
import heapq
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, train_test_split

target_list = ['Ga']        #'PD','SP',,'SP','Ga'
for target in target_list:
    best_R2 = [[],[],[],[],[]]
    save_R2 = [[],[],[],[],[]]
    data = pd.read_excel(f'5-1excels/{target}_merge_data.xlsx')
    y = data['nor']
    feature1 = data['hsv_h_median']                
    feature2 = data['hsv_s_median']
    feature3 = data['hsv_s_percentile_75']
    X = np.column_stack((feature1,feature2,feature3))
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    regressor = make_pipeline(PolynomialFeatures(2), LinearRegression())
    regressor.fit(X,y)
    #score = regressor.score(X_test, y_test)
    #print('Score: ', score)
    #print('Accuracy: ' + str(score*100) + '%')

    pred_data = pd.read_excel('2stage/piecesGA_color_space.xlsx')
    names = pred_data['name']
    f1 = pred_data['hsv_h_median']                
    f2 = pred_data['hsv_s_median']
    f3 = pred_data['hsv_s_percentile_75']
    Z = np.column_stack((f1,f2,f3))
    y_pred = regressor.predict(Z)
    for index, name in enumerate(names):
        print(f'{name}: {y_pred[index]}')
    df = pd.DataFrame({'name':names, 'pred':y_pred})
    file_path = f'./test/{target}_pred.xlsx'
    with pd.ExcelWriter(file_path, engine = 'openpyxl', mode = 'w') as writer:
        df.to_excel(writer, sheet_name=target, index = False)
        print(f'{target} predict saved.')
    