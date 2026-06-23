# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import matplotlib.pyplot as plt

# 解决matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================1、读取半导体数据集======================
df = pd.read_csv("../data/semiconductor.csv", encoding="utf-8")
X = df[["lattice_a", "atom_Z", "bond_energy", "density"]]
y = df["Eg"]

# ======================2、划分训练集70%、测试集30%======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

# ======================3、训练多元线性回归模型======================
model = LinearRegression()
model.fit(X_train, y_train)

# ======================4、输出回归系数、截距、完整预测公式======================
print("===============半导体禁带宽度多元线性回归计算结果===============")
b0 = model.intercept_
coef = model.coef_
print(f"回归截距 b0 = {b0:.4f}")
print(f"晶格常数系数  = {coef[0]:.4f}")
print(f"平均原子序数系数 = {coef[1]:.4f}")
print(f"平均键能系数 = {coef[2]:.4f}")
print(f"密度系数 = {coef[3]:.4f}")

formula = (
    f"\n预测计算公式：\n"
    f"Eg = {b0:.4f} + {coef[0]:.4f}×晶格常数 + {coef[1]:.4f}×平均原子序数 + "
    f"{coef[2]:.4f}×平均键能 + {coef[3]:.4f}×密度"
)
print(formula)

# ======================5、模型拟合效果评价======================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print("\n===============模型精度评价指标===============")
print(f"训练集决定系数 R² = {r2_score(y_train, y_train_pred):.4f}")
print(f"测试集决定系数 R² = {r2_score(y_test, y_test_pred):.4f}")
print(f"测试集平均绝对误差 MAE = {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"测试集均方误差 MSE = {mean_squared_error(y_test, y_test_pred):.4f}")

# ======================6、创建result文件夹，保存模型与文字结果======================
if not os.path.exists("../result"):
    os.mkdir("../result")
joblib.dump(model, "../result/eg_model.pkl")
with open("../result/回归结果.txt", "w", encoding="utf-8") as f:
    f.write("半导体禁带宽度多元线性回归模型\n")
    f.write(f"截距b0：{b0:.4f}\n")
    f.write(f"晶格常数系数：{coef[0]:.4f}\n")
    f.write(f"平均原子序数系数：{coef[1]:.4f}\n")
    f.write(f"平均键能系数：{coef[2]:.4f}\n")
    f.write(f"密度系数：{coef[3]:.4f}\n")
    f.write(formula + "\n\n")
    f.write("模型评价指标\n")
    f.write(f"训练集R²：{r2_score(y_train, y_train_pred):.4f}\n")
    f.write(f"测试集R²：{r2_score(y_test, y_test_pred):.4f}\n")
    f.write(f"测试集MAE：{mean_absolute_error(y_test, y_test_pred):.4f}\n")
    f.write(f"测试集MSE：{mean_squared_error(y_test, y_test_pred):.4f}\n")

# ======================7、单种材料自定义参数预测函数======================
def predict_Eg(lattice_a, atom_Z, bond_energy, density):
    res = model.predict([[lattice_a, atom_Z, bond_energy, density]])
    return round(res[0],4)

print("\n===============单材料预测演示(GaAs)===============")
Eg_GaAs = predict_Eg(5.653,32,3.62,5.317)
print(f"GaAs材料预测禁带宽度 Eg = {Eg_GaAs} eV")

# ======================8、全部材料批量预测======================
print("\n===============全部材料批量预测结果===============")
batch_data = [
    {"name":"Si","lattice_a":5.431,"atom_Z":14,"bond_energy":2.32,"density":2.33},
    {"name":"Ge","lattice_a":5.658,"atom_Z":32,"bond_energy":2.12,"density":5.32},
    {"name":"GaAs","lattice_a":5.653,"atom_Z":32,"bond_energy":3.62,"density":5.32},
    {"name":"GaP","lattice_a":5.451,"atom_Z":23,"bond_energy":3.48,"density":4.13},
    {"name":"GaN","lattice_a":3.189,"atom_Z":22,"bond_energy":4.27,"density":6.15},
    {"name":"InP","lattice_a":5.869,"atom_Z":41,"bond_energy":3.35,"density":4.81},
    {"name":"InAs","lattice_a":6.058,"atom_Z":42,"bond_energy":3.21,"density":5.67},
    {"name":"ZnO","lattice_a":3.249,"atom_Z":23,"bond_energy":3.72,"density":5.61},
    {"name":"ZnS","lattice_a":5.406,"atom_Z":23,"bond_energy":3.45,"density":4.09},
    {"name":"ZnSe","lattice_a":5.668,"atom_Z":32,"bond_energy":3.12,"density":5.27},
    {"name":"CdS","lattice_a":5.832,"atom_Z":32,"bond_energy":3.10,"density":4.82},
    {"name":"CdTe","lattice_a":6.481,"atom_Z":50,"bond_energy":2.62,"density":5.85},
    {"name":"TiO2","lattice_a":3.785,"atom_Z":18,"bond_energy":4.05,"density":3.89}
]
df_batch = pd.DataFrame(batch_data)
X_batch = df_batch[["lattice_a","atom_Z","bond_energy","density"]]
df_batch["预测Eg(eV)"] = model.predict(X_batch)
print(df_batch[["name","预测Eg(eV)"]])
df_batch.to_csv("../result/全部材料预测结果.csv",index=False,encoding="utf-8-sig")
print("批量预测数据已保存至 result/全部材料预测结果.csv")

# ======================9、批量预测柱状对比图======================
plt.figure(figsize=(12,5.5))
plt.bar(df_batch["name"],df_batch["预测Eg(eV)"],color="#2E86AB",width=0.55)
plt.xlabel("半导体材料")
plt.ylabel("预测禁带宽度 Eg / eV")
plt.title("13种半导体材料禁带宽度预测结果对比")
plt.grid(axis="y",alpha=0.35,linestyle="--")
plt.tight_layout()
plt.savefig("../result/材料预测柱状图.png",dpi=300)
plt.close()

# ======================10、模型拟合散点评价图======================
plt.figure(figsize=(9,6))
plt.scatter(y_train,y_train_pred,c="#A23B72",label="训练集样本",s=45)
plt.scatter(y_test,y_test_pred,c="#F18F01",label="测试集样本",s=45)
lim_min = min(min(y),min(y_train_pred),min(y_test_pred))-0.15
lim_max = max(max(y),max(y_train_pred),max(y_test_pred))+0.15
plt.plot([lim_min,lim_max],[lim_min,lim_max],"k--",lw=1.8,label="理想拟合线 y=x")
plt.xlabel("实测禁带宽度 Eg / eV")
plt.ylabel("模型预测禁带宽度 Eg / eV")
plt.title("多元线性回归模型：实测值-预测值拟合效果")
plt.legend(loc="best")
plt.grid(alpha=0.3,linestyle="--")
plt.tight_layout()
plt.savefig("../result/模型拟合散点图.png",dpi=300)
plt.close()

print("\n✅全部运行完成，结果、图表、模型文件生成在result文件夹")