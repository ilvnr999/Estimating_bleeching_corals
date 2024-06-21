import pandas as pd

def main():
    # 讀取第一個Excel文件
    file_path_1 = 'stage2_excels/PD/PD_merge_data.xlsx'
    df1 = pd.read_excel(file_path_1)

    # 讀取第二個Excel文件
    file_path_2 = 'temporary/protien.xlsx'
    df2 = pd.read_excel(file_path_2)

    # 構建一個從第二個Excel文件中提取數據的字典
    replace_dict = pd.Series(df2.iloc[:, 1].values, index=df2.iloc[:, 0]).to_dict()

    # 定義一個函數，用於根據名稱匹配替換第三列的值
    def replace_value(row):
        if row.iloc[0] in replace_dict:
            return replace_dict[row.iloc[0]]
        return row.iloc[2]

    # 使用apply方法遍歷每一行並進行替換第三列的值
    df1.iloc[:, 2] = df1.apply(replace_value, axis=1)

    # 定義一個函數，用於根據名稱匹配計算並替換第四列的值
    def calculate_and_replace(row):
        if row.iloc[0] in replace_dict:
            return row.iloc[1] / replace_dict[row.iloc[0]]
        return row.iloc[3]

    # 使用apply方法遍歷每一行並計算替換第四列的值
    df1.iloc[:, 3] = df1.apply(calculate_and_replace, axis=1)

    # 將修改後的數據保存回原Excel文件
    df1.to_excel(file_path_1, index=False)

    print(f"第一個Excel文件中的數據已根據名稱匹配進行更新並保存到 {file_path_1}")


if __name__ == "__main__":
    # 呼叫 main 函數並傳入目標清單
    main(['PD', 'SP', 'GA'])