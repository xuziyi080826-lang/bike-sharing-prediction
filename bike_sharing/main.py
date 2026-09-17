import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1 读取数据
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}")
        return None


# 2 特征工程
def feature_process(df):
    """
    特征工程：标签编码季节、节假日、运营日；过滤非运营日样本；提取月份新特征
    :param df: 原始读取的DataFrame
    :return df_model: 完成特征处理后的数据集
    """
    season_map= {
         "Winter":0,
         "Spring":1,
        "Summer":2,
        "Autumn":3
    }
    df["Seasons_label"] = df["Seasons"].map(season_map)

    print("\n==== 季节编码后统计 ====")
    print(df["Seasons_label"].value_counts())

    # Holiday：No Holiday →0 ; Holiday →1
    holiday_map = {
        "No Holiday":0,
        "Holiday":1
    }
    df["Holiday_label"] = df["Holiday"].map(holiday_map)

    print("\n==== Holiday编码统计 ====")
    print(df["Holiday_label"].value_counts())

    # Functioning Day Yes=1 , No=0
    func_map = {
     "Yes":1,
     "No":0
    }
    df["FuncDay_label"] = df["Functioning Day"].map(func_map)

    print("\n==== FunctioningDay编码统计 ====")
    print(df["FuncDay_label"].value_counts())

    print("\n==== 过滤前总样本数：",len(df))
    # 只保留运营日 FuncDay_label ==1；.copy()避免SettingWithCopyWarning警告
    df_model = df[df["FuncDay_label"] == 1].copy()
    print("==== 过滤后建模样本数：",len(df_model))

    # Date字符串转换为时间日期格式
    df_model["Date"] = pd.to_datetime(df_model["Date"],format="%d/%m/%Y")
    # 提取月份作为新特征
    df_model["Month"] = df_model["Date"].dt.month

    print("\n==== 衍生特征：月份统计 ====")
    print(df_model["Month"].value_counts().sort_index())
    return df_model
#3 数据集划分
def split_dataset(df, feature_cols):
    """
    划分训练集、测试集，test_size=0.3，random_state=42保证实验可复现
    :param df: 经过特征工程处理后的数据集
    :param feature_cols: 用于建模的特征列名称列表
    :return X_train,X_test,y_train,y_test: 划分好的特征与标签
    """
    X = df[feature_cols]
    y = df["Rented Bike Count"]
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.3,random_state=42
    )
    return X_train,X_test,y_train,y_test

#4 模型训练评估（已经改成随机森林）
def train_evaluate(X_train,X_test,y_train,y_test):
    """
    随机森林回归模型训练，在测试集预测，计算MAE、RMSE评估指标
    :param X_train:训练集特征
    :param X_test:测试集特征
    :param y_train:训练集标签
    :param y_test:测试集标签
    :return model,y_pred:训练完成的模型、测试集预测结果
    """
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test,y_pred)
    rmse = root_mean_squared_error(y_test,y_pred)
    print(f"MAE:{mae:.2f}")
    print(f"RMSE:{rmse:.2f}")
    return model,y_pred

# 程序入口，自动串联全部流程
if __name__ == "__main__":
    df_raw = load_data("BikeData.csv")
    df_feature = feature_process(df_raw)
   
# 建模使用的输入特征集合
    feature_cols = [
        "Hour","Temperature(°C)","Humidity(%)","Wind speed (m/s)",
        "Rainfall(mm)","Snowfall (cm)","Seasons_label","Holiday_label","Month",
    ]
    X_train, X_test, y_train, y_test = split_dataset(df_feature, feature_cols)
    model, y_pred = train_evaluate(X_train, X_test, y_train, y_test)
    # Day5 可视化1：真实值、预测值对比图
    plt.figure(figsize=(12,6))
    plt.plot(y_test.values[:200],label="真实租赁数量")
    plt.plot(y_pred[:200],label="预测租赁数量")
    plt.title("共享单车真实值 VS 预测值对比（前200条测试数据）")
    plt.xlabel("样本序号")
    plt.ylabel("租赁数量")
    plt.legend()
    plt.tight_layout()
    plt.savefig("true_pred_line.png")


    # Day5 可视化2：残差分布分析
    residual = y_test.values - y_pred

    plt.figure(figsize=(10,5))
    plt.hist(residual,bins=50)
    plt.title("模型残差分布（误差分布）")
    plt.xlabel("残差大小")
    plt.ylabel("样本数量")
    plt.tight_layout()
    plt.savefig("residual_hist.png")

    # Day5 可视化3：拟合散点图
    plt.figure(figsize=(8,6))
    plt.scatter(y_test,y_pred,alpha=0.6)
    plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color="red")
    plt.title("真实值‑预测值拟合散点图")
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.tight_layout()
    plt.savefig("scatter_fit.png")
    plt.show()

    print("\n===== 随机森林模型特征重要性 =====")
    for feat_name, importance in zip(feature_cols, model.feature_importances_):
        print(f"{feat_name} : {importance:.3f}")



# import pandas as pd
# import matplotlib.pyplot as plt

# # 解决matplotlib中文乱码 + 负号正常显示
# plt.rcParams["font.family"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False


# df = pd.read_csv(r"D:\code\bike_sharing\BikeData.csv")

# # Rented Bike Count：自行车租赁数量，为本数据集的预测目标
# print("===== 表格前5行 =====")
# print(df.head())

# print("\n===== 表格形状（行数，列数） =====")
# print(df.shape)


# print("\n===== 每一列的数据类型 =====")
# print(df.dtypes)


# print("\n===== 数字列统计信息 =====")
# print(df.describe())

# print("\n===== 所有列名称 =====")
# print(df.columns.tolist())


# print("\n===== Rented Bike Count（租赁数量）前10条 =====")
# print(df["Rented Bike Count"].head(10))

# winter_data = df[df["Seasons"] == "Winter"]
# print("\n===== 冬天(Winter)的样本条数 =====")
# print(len(winter_data))

# print("\n===== 每一列缺失值数量 =====")
# print(df.isnull().sum())

# print("\n===== 缺失值占每列总数据的百分比 =====")
# print(df.isnull().mean() * 100)

# print("\n===== Seasons列所有取值统计 =====")
# print(df["Seasons"].value_counts())

# print("\n===== Holiday列所有取值统计 =====")
# print(df["Holiday"].value_counts())

# print("\n===== Functioning Day列所有取值统计 =====")
# print(df["Functioning Day"].value_counts())


# # 全部转为首字母大写，其余小写，消除大小写脏数据
# df["Seasons"] = df["Seasons"].str.capitalize()
# df["Functioning Day"] = df["Functioning Day"].replace("Y", "Yes")

# print("\n===== 清洗之后Seasons分布 =====")
# print(df["Seasons"].value_counts())

# print("\n===== Rented Bike Count 统计 =====")
# print(df["Rented Bike Count"].describe())

# print("\n===== Temperature(°C) 统计 =====")
# print(df["Temperature(°C)"].describe())

# col = df["Rented Bike Count"]
# Q1 = col.quantile(0.25)
# Q3 = col.quantile(0.75)
# IQR = Q3 - Q1
# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR

# print(f"\nRented Bike Count 异常下界 {lower},异常上界 {upper}")
# outlier = df[(col < lower) | (col > upper)]
# print(f"检测到疑似异常样本数量：{len(outlier)}")

# # 输出清洗后的副本，不改动原始BikeData.csv
# df.to_csv(r"D:\code\bike_sharing\bike_cleaned.csv",index=False,encoding="utf-8")
# print("\n✅清洗完成，输出 bike_cleaned.csv")


# print("\n==== Day3：查看待转换的分类列全部取值 ====")
# print("Seasons：",df["Seasons"].unique())
# print("Holiday：",df["Holiday"].unique())
# print("Functioning Day：",df["Functioning Day"].unique())

# # 季节标签编码映射 Winter=0 Spring=1 Summer=2 Autumn=3
# season_map = {
#     "Winter":0,
#     "Spring":1,
#     "Summer":2,
#     "Autumn":3
# }
# df["Seasons_label"] = df["Seasons"].map(season_map)

# print("\n==== 季节编码后统计 ====")
# print(df["Seasons_label"].value_counts())

# # Holiday：No Holiday →0 ; Holiday →1
# holiday_map = {
#     "No Holiday":0,
#     "Holiday":1
# }
# df["Holiday_label"] = df["Holiday"].map(holiday_map)

# print("\n==== Holiday编码统计 ====")
# print(df["Holiday_label"].value_counts())

# # Functioning Day Yes=1 , No=0
# func_map = {
#     "Yes":1,
#     "No":0
# }
# df["FuncDay_label"] = df["Functioning Day"].map(func_map)

# print("\n==== FunctioningDay编码统计 ====")
# print(df["FuncDay_label"].value_counts())

# print("\n==== 过滤前总样本数：",len(df))
# # 只保留运营日 FuncDay_label ==1；.copy()避免SettingWithCopyWarning警告
# df_model = df[df["FuncDay_label"] == 1].copy()
# print("==== 过滤后建模样本数：",len(df_model))

# # Date字符串转换为时间日期格式
# df_model["Date"] = pd.to_datetime(df_model["Date"],format="%d/%m/%Y")
# # 提取月份作为新特征
# df_model["Month"] = df_model["Date"].dt.month

# print("\n==== 衍生特征：月份统计 ====")
# print(df_model["Month"].value_counts().sort_index())

# # 建模使用的输入特征集合
# feature_cols = [
#     "Hour","Temperature(°C)","Humidity(%)","Wind speed (m/s)",
#     "Rainfall(mm)","Snowfall (cm)","Seasons_label","Holiday_label","Month",
#    ]


# # 已经完成特征构造
# X = df_model[feature_cols]
# y = df_model["Rented Bike Count"]

# print("\n==== X前3行 ====")
# print(X.head(3))

# print("\n==== X特征矩阵形状(样本,特征数)：",X.shape)

# # 插入 train_test_split 切分代码
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.3,
#     random_state=42
# )

# # 打印切分后的形状
# print("训练集X_train形状：", X_train.shape)
# print("测试集X_test形状：", X_test.shape)
# print("训练集 y_train：",y_train.shape)
# print("测试集 y_test：",y_test.shape)

# df_model.to_csv("bike_feature_engineered.csv",index=False,encoding="utf-8")
# print("\n✅Day3完成，输出 bike_feature_engineered.csv")

# # ===================== Day4 模型训练部分 =====================
# #from sklearn.linear_model import LinearRegression

# # 创建模型对象
# #model = LinearRegression()
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(random_state=42)

# model.fit(X_train, y_train)

# # 在测试集做预测
# y_pred = model.predict(X_test)

# # 打印前10条真实值与预测值对照
# print("\n====真实值 vs 预测值 前10条====")
# for real_val, pred_val in zip(y_test[:10], y_pred[:10]):
#     print(f"真实:{real_val:.0f}  预测:{pred_val:.2f}")

# from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# mae = mean_absolute_error(y_test, y_pred)
# rmse = root_mean_squared_error(y_test, y_pred)

# print("\n====模型评估结果====")
# print(f"测试集 MAE = {mae:.2f}")
# print(f"测试集 RMSE = {rmse:.2f}")
# from sklearn.metrics import r2_score
# print("R² =", r2_score(y_test, y_pred))

# # Day5 可视化1：真实值、预测值对比图
# plt.figure(figsize=(12,6))
# plt.plot(y_test.values[:200],label="真实租赁数量")
# plt.plot(y_pred[:200],label="预测租赁数量")
# plt.title("共享单车真实值 VS 预测值对比（前200条测试数据）")
# plt.xlabel("样本序号")
# plt.ylabel("租赁数量")
# plt.legend()
# plt.tight_layout()
# plt.savefig("true_pred_line.png")


# # Day5 可视化2：残差分布分析
# residual = y_test.values - y_pred

# plt.figure(figsize=(10,5))
# plt.hist(residual,bins=50)
# plt.title("模型残差分布（误差分布）")
# plt.xlabel("残差大小")
# plt.ylabel("样本数量")
# plt.tight_layout()
# plt.savefig("residual_hist.png")

# # Day5 可视化3：拟合散点图
# plt.figure(figsize=(8,6))
# plt.scatter(y_test,y_pred,alpha=0.6)
# plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color="red")
# plt.title("真实值‑预测值拟合散点图")
# plt.xlabel("真实值")
# plt.ylabel("预测值")
# plt.tight_layout()
# plt.savefig("scatter_fit.png")
# plt.show()

# # Day5 模型解读：权重 + 偏置
# print("\n===== 随机森林模型特征重要性 =====")
# print("各特征重要性：")
# print(model.feature_importances_)


