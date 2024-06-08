import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import pinv
import pandas as pd
from sklearn.model_selection import KFold

class RBFNN:
    def __init__(self, input_dim, num_centers, output_dim):
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.output_dim = output_dim
        self.centers = None
        self.beta = None
        self.weights = None

    def _basis_function(self, center, data_point):
        return np.exp(-self.beta * np.linalg.norm(center - data_point) ** 2)

    def _calculate_interpolation_matrix(self, X):
        G = np.zeros((X.shape[0], self.num_centers))
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basis_function(c, x)
        return G

    def fit(self, X, Y):
        if self.num_centers <= 0 or self.num_centers > X.shape[0]:
            raise ValueError("Invalid number of centers")
        
        # Check if number of samples is at least equal to number of clusters
        if X.shape[0] < self.num_centers:
            raise ValueError("Number of samples should be greater than or equal to number of clusters")
        
        # Remove duplicate points
        X_unique, indices = np.unique(X, axis=0, return_index=True)
        Y_unique = Y[indices]

        # Check for infs or NaNs in X_unique and Y_unique
        if np.any(np.isnan(X_unique)) or np.any(np.isinf(X_unique)) or np.any(np.isnan(Y_unique)) or np.any(np.isinf(Y_unique)):
            raise ValueError("X_unique and Y_unique must not contain infs or NaNs")
        
        kmeans = KMeans(n_clusters=self.num_centers).fit(X_unique)
        self.centers = kmeans.cluster_centers_
        
        # Calculate beta parameter
        d_max = np.max([np.linalg.norm(c1 - c2) for c1 in self.centers for c2 in self.centers])
        self.beta = 1 / (2 * (d_max / np.sqrt(2 * self.num_centers)) ** 2)
        
        G = self._calculate_interpolation_matrix(X_unique)
        
        # Handle divide by zero
        if self.beta == 0:
            self.beta = 1e-6  # Use a small value instead of zero
        
        # Calculate weights
        self.weights = pinv(G).dot(Y_unique)
    def predict(self, X):
        G = self._calculate_interpolation_matrix(X)
        return G.dot(self.weights)
    
    def score(self, X, Y):
        Y_pred = self.predict(X)
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score

# 使用範例
if __name__ == "__main__":
    target_list = ['PD','SP','GA']
    for target in target_list:
        best_R2 = [[],[]]
        save_R2 = [[],[]]
        data = pd.read_excel(f'stage2_excels/{target}/{target}_merge_data.xlsx')
        nor = data.loc[:,'nor'].values
        all_tags = data.columns
        num_tags = len(all_tags)
        for i in range(4, num_tags):
            feature = data.iloc[:, i].values
            sort = np.argsort(feature)
            X = feature[sort].reshape(-1,1)
            x1 = tuple(feature)
            x2 = set(x1) 
            #print(x2)
            if len(x2)<=2:
                continue
            y = nor[sort]
            tag = all_tags[i]
            R2 = []
            #print(X)
            #print(y)
            rbf = RBFNN(input_dim=1, num_centers=2, output_dim=1)
            kfold = KFold(n_splits=5, shuffle=True, random_state=4)
            for train_index, test_index in kfold.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                rbf.fit(X_train,y_train)
                R2.append((rbf.score(X_test, y_test)))
                #print(rbf.score(X_test, y_test))
            best_R2[0].append(tag)
            best_R2[1].append(np.mean(R2))    
        print(target)
        best3 = np.array(best_R2[1])
        b = best3.argsort()
        length = len(b)
        for i in range(length-1,0,-1):
            print("%-20s %10.3f"%(best_R2[0][b[i]],best_R2[1][b[i]]))
            save_R2[0].append(best_R2[0][b[i]])
            save_R2[1].append(best_R2[1][b[i]])

        df = pd.DataFrame({"tag":save_R2[0], "R2":save_R2[1]})
        file_path = f'stage2_excels/{target}/{target}_kfold_RBFNN.xlsx'     # 輸出excel檔案名稱
        with pd.ExcelWriter(file_path, engine = 'openpyxl', mode = 'w') as writer:
            df.to_excel(writer, sheet_name=target, index = False)
            print(f'{target} kflod saved.')

