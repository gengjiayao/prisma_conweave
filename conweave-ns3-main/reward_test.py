import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 请替换为你的实际文件路径
csv_path = "/home/cuiyiqin/Cyq_RL/PRISMA_conweave/conweave-ns3-main/mix/output/2026.1.19解决了qSmooth贴地问题？/reward_breakdown_sw128.csv"

# 读取数据
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

# 过滤掉前 10% 的热身数据，避免噪音
warmup_idx = int(len(df) * 0.1)
df = df.iloc[warmup_idx:].copy()

# 确保数值类型
cols = ["T", "qSmooth", "q_score", "R_norm", "r_inst", "avg_queue_bytes", "bdp_bytes", "r_level"]
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("=== 1. 基础统计 ===")
print(df[cols].describe())

print("\n=== 2. 队列敏感度分析 ===")
# 计算队列占用相对于 BDP 的倍数
df['queue_usage_bdp'] = df['avg_queue_bytes'] / (df['bdp_bytes'] + 1e-9)
# 统计：当队列仅仅只有 1 倍 BDP 时，q_score 是多少？
mean_q_at_1bdp = df[(df['queue_usage_bdp'] > 0.8) & (df['queue_usage_bdp'] < 1.2)]['q_score'].mean()
print(f"Average q_score when Queue approx 1.0 * BDP: {mean_q_at_1bdp:.4f} (Expected ~0.5 if qRef=BDP)")
print(f"Average Queue Bytes: {df['avg_queue_bytes'].mean():.2f}")
print(f"Average BDP Bytes: {df['bdp_bytes'].mean():.2f}")

print("\n=== 3. 吞吐量(T)与奖励的矛盾分析 ===")
# 将 T 分为高低两组
t_high_thresh = df['T'].quantile(0.8)
t_low_thresh = df['T'].quantile(0.2)

high_t_group = df[df['T'] >= t_high_thresh]
low_t_group = df[df['T'] <= t_low_thresh]

print(f"High T Group (T >= {t_high_thresh:.2f}): Mean r_inst = {high_t_group['r_inst'].mean():.4f}")
print(f"Low T Group  (T <= {t_low_thresh:.2f}): Mean r_inst = {low_t_group['r_inst'].mean():.4f}")

diff = high_t_group['r_inst'].mean() - low_t_group['r_inst'].mean()
if diff < 0:
    print(f"结论: 高吞吐量导致奖励下降 {diff:.4f} -> Agent 被惩罚 working!")
else:
    print(f"结论: 高吞吐量带来奖励提升 {diff:.4f} -> Agent 被激励 working!")

print("\n=== 4. 相关性矩阵 (谁在拖后腿?) ===")
corr = df[['T', 'qSmooth', 'R_norm', 'r_inst']].corr()
print(corr)
print(f"\nCorr(T, qSmooth): {corr.loc['T', 'qSmooth']:.4f} (如果接近 -1，说明发包就导致队列分暴跌)")
print(f"Corr(T, R_norm):  {corr.loc['T', 'R_norm']:.4f} (如果接近 +1，说明发包就导致乱序惩罚)")

# (可选) 简单的绘图代码，如果你在本地运行可以生成图片
plt.figure(figsize=(10, 6))
plt.scatter(df['T'], df['r_inst'], alpha=0.5, s=2)
plt.xlabel('Throughput (T)')
plt.ylabel('Instant Reward (r_inst)')
plt.title('Throughput vs Reward')
plt.grid(True)
plt.savefig('t_vs_reward.png')
print("\nPlot saved to t_vs_reward.png")