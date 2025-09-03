import pandas as pd
import numpy as np

def calculate_errors(csv_file1, csv_file2, col_name1='boneage', col_name2='y_hat'):
    # 读取CSV文件
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    # 检查数据长度是否一致
    if len(df1) != len(df2):
        raise ValueError(f"两个CSV文件的数据行数不一致: {len(df1)} vs {len(df2)}")
    # 提取数据列
    data1 = df1[col_name1].values
    data2 = df2[col_name2].values
    # 计算误差
    errors = data1 - data2
    # 计算MAE
    mae = np.mean(np.abs(errors))
    # 计算RMSE
    rmse = np.sqrt(np.mean(errors **2))
    return mae, rmse
if __name__ == "__main__":
    # 示例用法
    file1 = "val_data.csv"
    file2 = "output/predictions_results.csv"

    try:
        mae, rmse = calculate_errors(file1, file2)
        print(f"两个CSV文件数据的平均绝对误差(MAD)为: {mae:.6f}")
        print(f"均方根误差(RMSE)为: {rmse:.6f}")
    except Exception as e:
        print(f"计算过程出错: {str(e)}")
