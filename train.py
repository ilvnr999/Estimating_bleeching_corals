import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import joblib
from openpyxl import Workbook

def train_and_save_model(target, X, y):
    R2 = []
    degree = 2
    if target == 'PS':
        degree = 3
    regressor = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    kfold = KFold(n_splits=5, shuffle=True, random_state=4)
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressor.fit(X_train, y_train)
        R2.append(regressor.score(X_test, y_test))
    
    R2_mean = np.mean(R2)
    model_filename = f'predict/{target}_linear_regression_model_run.pkl'
    joblib.dump(regressor, model_filename)
    print(f"模型已保存到 {model_filename}，R²: {R2_mean}")

    return R2_mean, model_filename

if __name__ == "__main__":
    target_list = ['PD', 'SP', 'GA']
    accuracy = []

    for target in target_list:
        data = pd.read_excel(f'stage2_excels/{target}/{target}_merge_data_rbf.xlsx')
        nor = data['nor']
        all_tags = data.columns
        columns_to_extract = data.iloc[:, [4, 5, 6]]

        columns_array = columns_to_extract.to_numpy()
        result = np.column_stack((columns_array[:, 0], columns_array[:, 1], columns_array[:, 2]))
        
        y = nor.to_numpy()
        X = result

        
        R2_mean, model_filename = train_and_save_model(target, X, y)
        accuracy.append({'target': target, 'R2': R2_mean, 'model': model_filename})

    # 将所有结果保存到 Excel 文件
    df = pd.DataFrame(accuracy)
    file_path = f'stage2_excels/RBFNN/kfold3.xlsx'
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)
        print(f'RBFNN3 kfold results saved.')
