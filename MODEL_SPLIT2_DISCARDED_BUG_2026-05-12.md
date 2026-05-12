# 致命 Bug：Q 网络丢弃了 40 维 CONGA 特征（2026-05-12）

> 严重性：🔴 致命 —— 解释了过去 5 个月所有"reward 改不好"的现象
> 触发：T_port 改为 cost 后 Agent 仍学反方向，深入审计模型架构发现

---

## 一、Bug 描述

`prisma/source/models.py:DQN_buffer_model` 在网络结构中**丢弃了 split[2] 的 40 维 CONGA 端口特征**。

### 原代码（错误）

```python
inp = layers.Input(shape=(observation_shape[0],))
split = SplitLayer(num_or_size_splits=input_size_splits)(inp)
# input_size_splits = [1, 1, 40]
# split[0] = dstOverlay   (1 维)
# split[1] = lastAction   (1 维)
# split[2] = CONGA 40 维  (8 ports × 5 features)

first_block  = one_hot(split[0]) → Dense(16)    # 用了 split[0] ✓
second_block = LayerNorm(split[1]) → Dense(16)  # 用了 split[1] ✓
                                                 # split[2] 从未被使用！❌

tensors_2_concat = [first_block, second_block]  # 只 concat 这俩
out = Dense(32) → Dense(32) → Dense(num_actions)
```

## 二、为什么这是过去 5 个月的根本症结

Q 网络真正看到的输入只有 **17 维**（16 维 one-hot dstOverlay + 1 维 lastAction），**40 维 CONGA 拥塞特征完全没进网络**。

这等于让 Agent 用一个 `lookup_table[dstOverlay][lastAction] → action` 的查表函数做路由决策——**完全脱离实际网络拥塞状态**。

### 一次性解释之前所有现象

| 历史现象 | 真实原因 |
|---------|---------|
| Reward 一直锁在低值 | Q 网络看不到端口状态，无法学习好坏 |
| 修 obs normalize 后 SNR 提升但 Agent 仍学反 | obs[2:] 根本没进网络，缩放怎么改都没用 |
| T_port 时间尺度错位 | 即使修对 T，T 也进不到 Q 网络 |
| Per-port reward 与 FCT 不对齐 | 不是 reward 设计问题，是 obs 输入残缺 |
| T_port 当 cost 后 mean_r 反向但 Agent 学反 | Q 网络只看 dst+lastAction，学不到拥塞依据 |
| FCT 输 ECMP 10-20% | 用 17 维信息做 8 端口选择 vs ECMP 用哈希——结果差不多 |

### 之前"reward 设计层面"的判断需要部分撤回

我之前推论：
- T_port 是 CONGA cost 用反了 → 部分正确，方向是反的
- Per-port reward 不能驱动 FCT → 这个推论可能不成立

修复 split[2] 后，需要重新评估这些判断。

## 三、修复

`prisma/source/models.py:DQN_buffer_model` 在 second_block 之后加入：

```python
tensors_2_concat = [first_block, second_block]
if len(input_size_splits) >= 3:
    third_block = layers.LayerNormalization(center=False, scale=False, trainable=False, axis=1)(split[2])
    third_block = layers.Dense(units=32, activation="elu", kernel_initializer='he_uniform', bias_initializer='he_uniform')(third_block)
    tensors_2_concat.append(third_block)
```

把 40 维 CONGA 特征经 LayerNorm + Dense(32) 后接入 Concatenate。

## 四、其他 model 检查

`models.py` 中其他 model（`dqn_buffer_lite/lighter/ff` 等）**都用了 for 循环遍历所有 split**，没这个问题。**只有 `DQN_buffer_model` 一个有此 bug**——这恰好是我们一直在用的 agent_type。

## 五、验证方法

跑短训（simul_time=0.5）后看：
- SNR 是否提升（之前 0.4~1.1 → 期望 > 2）
- 动作分布与 reward / T_port 是否强相关
- Q 网络输出的 8 个动作 Q 值差距是否明显

## 六、后续可能需要重新评估的点

1. **T_port 当 reward 还是 cost**？修 split[2] 后 Agent 真正看到端口状态，可能 T 当 reward 反而对了（"哪个端口在传数据 = 好选择"）。需要短训验证。

2. **reward 各分量权重**？现在 Agent 看得到了，可能需要重新调权重让信号更敏感。

3. **训练样本规模**？17 维输入下 Agent 学不到东西，所以"训练时长"问题被掩盖了。现在 57 维有效输入，DQN 可能需要更多样本才能收敛——之前每 leaf 33K 样本可能远不够。

## 七、本次修复的方法论价值

这个 bug 极其隐蔽：
- obs 维度对（42 维确实进了网络）
- SplitLayer 切分对（输出 3 个 tensor）
- 数值流通畅（split[2] tensor 存在）
- 但**没人把 split[2] 喂给后续 Dense 层**——split[2] 是个"孤儿 tensor"，TF 计算图里存在但不参与最终输出

只有深入审计模型架构每一行代码才能发现。这条经验教训：

> **数值正确 + 形状正确 + Loss 收敛 ≠ 网络真正用到了输入**
> 必须确认每一个输入维度都有从 Input 到 Output 的数据流路径

---

## 八、立即验证命令

短训（约 30 分钟）：

```bash
python3 main.py --train=1 --netload=50 --buffer=50 --simul_time=0.5 \
  --exploration_schedule_timesteps=8000 --exploration_final_eps=0.05 \
  --replay_buffer_max_size=100000 --pfc=0 --irn=1 --cc=dctcp \
  --session_name=split2_fixed_v1 --gamma=0.9
```
