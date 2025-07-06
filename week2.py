# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import rcParams
#设置字体为支持中文
rcParams['font.sans-serif'] = ['SimHei']  #指定默认字体为 SimHei
rcParams['axes.unicode_minus'] = False  #显示负号
# 数据加载
file_path = 'US-pumpkins.csv'  # 请根据实际路径调整
try:
    data = pd.read_csv(file_path)
except Exception as e:
    print(f"读取CSV文件时出错: {e}")
    try:
        data = pd.read_excel(file_path)
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        raise  # 如果两种方式都失败，抛出异常

# 数据基本信息查看
print("数据基本信息：")
data_info = data.info()
print("\n各列缺失值数量：")
missing_values = data.isnull().sum()
print(missing_values)

# 数据清洗
# 删除全部为缺失值的列
columns_to_drop = data.columns[data.isnull().all()].tolist()
data_cleaned = data.drop(columns=columns_to_drop)
# 选择关键列，并删除这些关键列中存在缺失值的行
critical_columns = ['City Name', 'Package', 'Variety', 'Date', 'Low Price', 'High Price']
data_cleaned = data_cleaned.dropna(subset=critical_columns)
# 将 'Date' 列转换为 datetime 格式
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], errors='coerce')
# 再次检查数据信息
print("\n清洗后数据基本信息：")
data_cleaned_info = data_cleaned.info()
print("\n清洗后各列缺失值数量：")
data_cleaned_missing_values = data_cleaned.isnull().sum()
print(data_cleaned_missing_values)

# 探索性数据分析 (EDA)
# 绘制数值变量的分布直方图
numeric_columns = data_cleaned.select_dtypes(include=['float64']).columns
plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data_cleaned[column], kde=True)
    plt.title(f'分布 of {column}')
plt.tight_layout()
plt.show()

# 绘制数值变量的箱线图以识别异常值
plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=data_cleaned[column])
    plt.title(f'箱线图 of {column}')
plt.tight_layout()
plt.show()

# 绘制数值变量之间的相关性热力图
correlation_matrix = data_cleaned[numeric_columns].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('相关性热力图')
plt.show()

# 特征处理与模型构建
# 选择特征：选择缺失值少于50%的列作为特征
threshold = len(data_cleaned) * 0.5
features_with_few_missing = data_cleaned.columns[data_cleaned.isnull().sum() < threshold]
# 选择目标变量
target_variable = 'Low Price'
# 选择特征和目标变量
features_and_target = data_cleaned[features_with_few_missing.union([target_variable])]
# 删除所选特征和目标变量中存在缺失值的行
features_and_target = features_and_target.dropna(subset=features_with_few_missing.union([target_variable]))
# 对分类变量进行独热编码
categorical_columns = features_and_target.select_dtypes(include=['object']).columns
features_and_target_encoded = pd.get_dummies(features_and_target, columns=categorical_columns)
# 分离特征和目标变量
X = features_and_target_encoded.drop(columns=[target_variable])
y = features_and_target_encoded[target_variable]
# 检查是否存在 datetime 类型的列，并移除它们
datetime_columns = X.select_dtypes(include=['datetime64']).columns
if not datetime_columns.empty:
    X = X.drop(columns=datetime_columns)
    print(f"已移除 datetime 类型的列: {datetime_columns}")
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 初始化线性回归模型
model = LinearRegression()
# 训练模型
model.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = model.predict(X_test)
# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n模型评估结果：")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 绘制特征热力图
# 计算特征之间的相关性矩阵
feature_correlation_matrix = X.corr()

# 绘制热力图
plt.figure(figsize=(12, 10))
sns.heatmap(feature_correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('特征热力图')
plt.show()