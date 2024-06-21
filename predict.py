import numpy as np
import pandas as pd
import joblib

# 加载模型并进行预测
def main(target_list):
    for a in target_list:
        try:
            # 加载模型
            model_filename = f'predict/{a}_linear_regression_model_run.pkl'
            loaded_model = joblib.load(model_filename)

            # 读取输入文件
            input_file_path = f'predict/excels/{a}/{a}_predictions.xlsx'
            df = pd.read_excel(input_file_path)

            # 調試訊息：打印DataFrame的結構
            print(f"处理 {a} 的输入文件，DataFrame形状: {df.shape}")
            print(df.head())

            # 檢查DataFrame的列數
            if df.shape[1] < 4:
                raise ValueError(f"DataFrame中的列數不足4列，实际列数为 {df.shape[1]}")

            # 准备新的 DataFrame 来保存结果
            result_df = df.iloc[:, [0]].copy()  # 复制第一列（名称列）

            # 提取第2到第4列数据并堆叠在一起
            X_new = df.iloc[:, 1:4].values
            print(f"提取的输入特征形状: {X_new.shape}")

            # 確認選取的列數是否正確
            if X_new.shape[1] != 3:
                raise ValueError(f"提取的特徵數量不正確，預期3個，實際為 {X_new.shape[1]}")

            # 进行预测
            predictions = loaded_model.predict(X_new)

            # 将预测结果添加到新列
            result_df['Prediction'] = predictions
            
            # 保存结果到新的 Excel 文件
            output_file_path = f'predict/excels/{a}/{a}stage2_predictions.xlsx'
            result_df.to_excel(output_file_path, index=False)

            print(f"预测结果已保存到 {output_file_path}")
        
        except Exception as e:
            print(f"处理 {a} 时发生错误: {e}")

if __name__ == "__main__":
    # 呼叫 main 函數並傳入目標清單
    main(['PD','SP','GA'])
