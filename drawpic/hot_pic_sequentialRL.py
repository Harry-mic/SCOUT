import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局风格
sns.set_theme(style="whitegrid")

# --- 数据定义 ---
tasks = ['Bandit', 'FL_Static', 'FL_Slippery', 'S_Box1', 'S_Box2', 'R_Rot1', 'R_Rot2', 'R_Rot3', 'Sudoku']

# SCOUT 数据
scout_data = [
    [0.77, 0.24, 0.33, 0.13, 0.02, 0.14, 0.04, 0.04, 0.00],
    [1.00, 0.91, 0.87, 0.46, 0.15, 1.00, 1.00, 0.88, 0.38],
    [1.00, 0.91, 0.86, 0.46, 0.15, 1.00, 1.00, 0.88, 0.40],
    [1.00, 0.93, 0.90, 0.50, 0.15, 1.00, 1.00, 0.88, 0.43],
    [1.00, 0.89, 0.88, 0.93, 0.59, 1.00, 1.00, 0.86, 0.48],
    [1.00, 0.89, 0.88, 0.95, 0.59, 1.00, 1.00, 0.88, 0.52],
    [1.00, 0.89, 0.88, 0.95, 0.59, 1.00, 1.00, 0.89, 0.98]
]
scout_stages = ['Base', '+SFT', '+Bandit', '+Frozenlake', '+Sokoban', '+Rubiks', '+Sudoku']
df_scout = pd.DataFrame(scout_data, columns=tasks, index=scout_stages)

# Direct RL 数据
direct_data = [
    [0.77, 0.24, 0.33, 0.13, 0.02, 0.14, 0.04, 0.04, 0.00],
    [0.86, 0.26, 0.25, 0.14, 0.02, 0.11, 0.04, 0.04, 0.00],
    [0.84, 0.22, 0.30, 0.17, 0.06, 0.23, 0.05, 0.08, 0.00],
    [0.82, 0.51, 0.39, 0.40, 0.10, 0.17, 0.10, 0.07, 0.00],
    [0.70, 0.52, 0.50, 0.37, 0.09, 0.33, 0.18, 0.11, 0.02],
    [0.80, 0.59, 0.48, 0.34, 0.10, 0.33, 0.22, 0.11, 0.34]
]
direct_stages = ['Base', '+Bandit', '+Frozenlake', '+Sokoban', '+Rubiks', '+Sudoku']
df_direct = pd.DataFrame(direct_data, columns=tasks, index=direct_stages)


# --- 绘制图 1: SCOUT ---
plt.figure(figsize=(10, 6)) # 单独的画布大小

sns.heatmap(df_scout, 
            annot=True, 
            fmt=".2f", 
            cmap="YlGnBu", # 建议使用一致的色系
            vmin=0, vmax=1, # 【重要】锁定范围，确保两张图颜色可比
            cbar=True,      # 单独画时，需要开启色条
            linewidths=.5)

plt.title('SCOUT Sequential RL: Task Performance Evolution', fontsize=14, pad=15)
plt.ylabel('Training Stage', fontsize=12)
plt.xlabel('Task', fontsize=12)
plt.tight_layout()

# 保存
plt.savefig('/mnt/general/wanghy/RAGEN/drawpic/save/heatmap_scout.png', dpi=300, bbox_inches='tight')
plt.show()