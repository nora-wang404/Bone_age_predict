import os.path

import pandas as pd
import numpy as np

from data_process.data_config import rootdata


def split_csv(input_file, train_output, test_output, split_ratio=0.8):

    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        print(f"成功读取文件: {input_file}，共 {len(df)} 行数据")

        # 打乱行顺序
        # 使用.sample()方法并指定frac=1表示返回所有行的随机排列
        df_shuffled = df.sample(frac=1, random_state=np.random.randint(0, 1000))

        # 计算分割点
        split_index = int(len(df_shuffled) * split_ratio)

        # 分割数据
        df_train = df_shuffled[:split_index]
        df_test = df_shuffled[split_index:]

        # 保存为CSV文件
        df_train.to_csv(train_output, index=False)
        df_test.to_csv(test_output, index=False)

        print(f"数据分割完成:")
        print(f"训练集: {len(df_train)} 行，已保存至 {train_output}")
        print(f"测试集: {len(df_test)} 行，已保存至 {test_output}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    # 示例用法
    input_csv = os.path.join(rootdata,"boneage-training-dataset.csv")
    train_csv = os.path.join(rootdata,"train_data.csv")  # 训练集输出文件
    val_csv = os.path.join(rootdata,"val_data.csv") # val集输出文件

    split_csv(input_csv, train_csv, val_csv)
