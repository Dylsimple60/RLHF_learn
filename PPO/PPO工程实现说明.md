# PPO工程实现
![image.png](PPO%E5%B7%A5%E7%A8%8B%E5%AE%9E%E7%8E%B0/image.png)

## 先验理论

### 1. 我们在干什么：最大化期望回报（Policy Gradient）

我们有一个参数化策略：$\pi_\theta(a|s)$

目标是让它在环境中获得尽量大的累计奖励：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
,\quad
R(\tau)=\sum_{t=0}^{T}\gamma^t r_t
$$

**策略梯度定理**告诉我们：$\nabla_\theta J(\theta)
=\mathbb{E}[G_t \nabla_\theta\log\pi_\theta(a_t|s_t)]$

所以最朴素的更新式（REINFORCE）是： $\theta \leftarrow \theta + \alpha G_t \nabla_\theta\log\pi_\theta(a_t|s_t)$

### 2. 纯 Policy Gradient 的核心问题

1. **高方差： $G_t$** 是整段未来回报，波动巨大 → 更新抖得厉害
2. **样本利用率低：**每次更新后策略变了，旧数据就“过期”了 → 只能用一次
3. **更新不稳定：**大步更新可能把策略一下子推到一个很差的地方，甚至崩掉

$G_t$解决整体思路： a.$G_t$太难估计b.$A_t=G_t-V(s)$方差太大c.采用多步 TD 信息估计 $A_t$

### 3. Actor–Critic：第一步改进

核心思想：

- 加一个 **Critic（值函数 V(s) 或 Q(s,a)）** 来估计回报
- 用 **Advantage Aₜ** 替换 $G_t$，降低方差

典型形式： $\quad
A_t= Q(s_t,a_t) - V(s_t)\approx G_t - V(s_t)$

$A_t = \underbrace{\text{动作选择后的实际表现}}_{Q(s,a)} 
- \underbrace{\text{在该状态下的平均表现}}_{V(s)}$

更新变成： $\theta \leftarrow \theta + \alpha A_t\nabla_\theta\log\pi_\theta(a_t|s_t)$

**动机：**

- V(s) 作为 baseline，把“环境本身好坏”的成分减掉
- 只关注“这次动作比平常水平好多少”，偏移显著降低
- Critic 用 TD-learning 每步更新，比 MC 要稳定

但仍有两个问题没解决好：

1. 策略更新步子可能太大 → 不够稳定（用Clip解决）
2. 旧数据还不能多次使用 → 样本效率一般


### 4.Clip：限制策略更新幅度

### Clip 目标函数（最核心的 PPO 公式）：

$$
L^{clip}(\theta) = \mathbb{E}_t \left[
\min(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)
\right]
$$

解释：

- **$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，概率比,策略更新前后，对同一个动作的概率变化了多少；如果 $r_t$ 太大太小，就强制剪裁到 [1−ε,1+ε]**
- 用 `min` 强制更新不要超过剪裁后的界限

### 5.GAE：广义优势估计（Generalized Advantage Estimation）

在 Actor–Critic 中我们使用： $A_t = G_t - V(s_t)$，但这是MC方式估计，**方差很大**（因为包含整段未来），TD方式估计 → r + γV(s’) - V(s)，偏差较大。

GAE 解决方式：综合了多步 TD 信息来计算**Advantage Aₜ** （整体思路）

---

 GAE 的基本形式：

$$
A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l 
\delta_{t+l}=
\delta_t 
+ (\gamma\lambda)\delta_{t+1}
+ (\gamma\lambda)^2\delta_{t+2}
+ \cdots
$$

其中TD 残差： $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

- γ 控制长期性
- λ 控制偏差 vs 方差

**λ=1 → 更像 MC（低偏差、高方差）**

**λ=0 → 更像 TD（高偏差、低方差）**

GAE 的优点：

- 估计更稳定
- 适合 policy gradient 使用
- 几乎是所有现代 RL 算法都在用

### 6.优化目标

Actor目标：**最大化 PPO Clip 目标 $L^{clip}(\theta)=
\mathbb{E}[\min(r_tA_t,\;\text{clip}(r_t)A_t)]$**

Critic目标：拟合价值函数 $L^{value}(\theta) = \left(V_\theta(s_t) - V^{target}_t\right)^2= (V_\theta(s_t) - Return)^2$

PPO 总目标（Actor–Critic 联合目标）： $L(\theta) = 
\mathbb{E}[L^{clip}(\theta)
- c_1 L^{value}(\theta)
+ c_2 S(\pi_\theta)]$

- $L^{clip}(\theta)$：策略目标，保持更新稳定
- $L^{value}(\theta)$：价值函数（Critic）损失
- $S(\pi_\theta)$：熵，促进探索
- $c_1,c_2$：权重系数

---

### 7. PPO 的整体训练流程（总结串接）

结合你前面提到的 Actor–Critic 以及上述内容，我们可以把 PPO 的逻辑串起来：

---

**(1) 与环境交互：** 收集一批 trajectories：$(s_t, a_t, r_t)$

---

**(2) 用 Critic 估计价值并计算 GAE 优势：**$A_t^{GAE} = \text{GAE}(\delta_t)$

---

**(3) 记录旧的策略概率：**$\pi_{\theta_{old}}$

---

**(4) 进行若干 epoch 的更新（体现“样本可重复利用”）**

对 Actor：$L^{clip}(\theta)$

对 Critic：$L^{value}(\theta)$

加上熵：$S(\pi_\theta)$

---

**(5) 通过 min(clip, unclip) 强制更新幅度受限：** 确保策略不会离旧策略太远。

---

**(6) 循环 1~5，持续训练：** 最终得到一个稳定、样本效率高、性能优异的策略。

## 基于PPO算法的大语言模型(LLM)强化学习训练代码

![image.png](PPO%E5%B7%A5%E7%A8%8B%E5%AE%9E%E7%8E%B0/image%201.png)

### 四个模型：策略模型、价值模型、奖励模型、参考模型

策略模型：待优化的模型，参与参数更新

价值模型：计算当前动作和状态的期望回报，可由奖励模型或策略模型初始化而来，参与参数更新

奖励模型：计算当前动作的即时奖励，不参与参数更新

参考模型：由策略模型进行初始化，不参与参数更新，用于限制策略模型在优化时不偏离原始模型太远（用参考模型和策略模型来计算KL散度）

### 两个损失：策略损失、价值损失

策略损失：用于优化策略模型

价值损失：用于优化价值模型

**huggingface中模型如何查看git地址**

![image.png](PPO%E5%B7%A5%E7%A8%8B%E5%AE%9E%E7%8E%B0/image%202.png)

**huggingface拉取不下来用AutoDL学术资源加速：**[https://www.autodl.com/docs/network_turbo/](https://www.autodl.com/docs/network_turbo/)

**reward-model-deberta-v3-large-v2：**给定一个QA对，能够给出一个分数来衡量QA对的质量。

### 整体伪代码：LLM + PPO

**1.模型设置**

```python
# 策略模型 
actor_model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct').to(device)
# 参考模型
ref_model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct').to(device)
# 奖励模型
reward_model = AutoModelForSequenceClassification.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2').to(device)
# 价值模型
critic_model = Critic(actor_model.base_model).to(device)
```

**2.准备 Prompt 数据集**（问题列表）

把原始文本 prompt（如 “请问1+1等于多少？”）转换成：LLM 聊天模板格式

```python
输入：请问1+1等于多少？
输出：<bos><user> 请问1+1等于多少？ </user><assistant>
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer = tokenizer
        
        self.final_prompts = []
        
        for prompt in prompts:
            if apply_chat_template:
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
            else:
                prompt = self.tokenizer.bos_token + prompt
                
            self.final_prompts.append(prompt)
```

1. **rollout：策略模型对 prompt 生成回答（model.generate）**

对每条 prompt，策略模型 πθ 生成回答：

attention_mask 的作用：告诉模型：哪些位置是有效 token，哪些只是 padding，不参与注意力、loss。

action_mask：action_mask 只针对 **回答部分**。是回答的 token → 1；是 EOS 或 padding → 0

num_actions = 回答（answer）部分的 token 数量；只对这几个 token 计算 logprob、 KL、advantage、PPO loss

```python
输入：<bos><user> 请问1+1等于多少？ </user><assistant>
模型回答为“2 <eos>”
输出：
Samples(
            seqs=seqs,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),
            packed_seq_lens=None,
            response_length=action_mask.float().sum(dim=-1),  
            total_length=attention_mask.float().sum(dim=-1), （prompt + 回答 的总有效长度）
        )
 seqs = [
    1,10,101,102,103,104,105,106,107,108,109,110,11,12,0,0,  ← prompt 区（对应的内容是请问1+1等于多少？）
    120, 2, 0, 0, 0, 0, ...                                  ← answer 区 (max_new_tokens=6)（2 <eos>）
]
attention_mask = [
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,   ← prompt
    1,1,0,0,0,0                         ← answer
]
action_mask = [0,0,0,...,0,   1, 0, 0,0,0,0]  （只有120（对应汉字2）被标记为1）
num_actions = 1

```

1. **形成 Experience，计算：logprob、KL、reward、value、advantage、return**

```python
输入：Samples 对象
输出：Experience 对象

a. 计算actor模型输出token的概率
对应的 action_log_probs：[-0.1, -5.0, -20, -20, -20, -20]
解释：回答部分为answer tokens：["2", "<eos>", pad, pad, pad, pad]，
则对应的logprob ≈ -0.1  →  概率 p ≈ exp(-0.1) ≈ 0.90；logprob ≈ -5  →  概率p ≈ exp(-5) ≈ 0.0067
说明“2” 的 logprob 较高（模型自信）
b. 计算参考actor模型输出token的概率
ref_action_log_probs = [-0.2, -4.5, -20, -20, -20, -20]
c. KL 惩罚(KL ≈ logπθ - logπref)
kl = [0.1, -0.5, 0, 0, 0, 0]
d.奖励模型得分
r_rm = [1.5]
e.用 GAE 计算 advantage A_t 和 return G_t

experiences.append(Experience(seqs,
            action_log_probs.detach(),
            value.detach(),
            returns.detach(),
            advantages.detach(),
            attention_mask,
            action_mask,
            r.detach(),
            samples.response_length,
            samples.total_length,
            num_actions,
            kl.detach(),
))

```

**明明我喂给 actor_model 整个 seqs，它怎么就给出了像 [-0.1, -5.0, -20, ...] 这样的 “answer token 概率（logprob）” 呢？**

```python
# 计算策略模型输出token的概率
#拿整条 token 序列 seqs 喂给策略模型，做一次前向计算，得到每个位置对“下一个 token”的预测结果（logits 等）
output = actor_model(seqs, attention_mask=attention_mask)
#每个位置产生一组 logits，形状 [batch_size, seq_len, vocab_size]
logits = output.logits
#每个位置上词表中每个 token 的 log 概率
log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
#再用 gather 取出“当前序列里真实 token 的 logprob”
log_probs_labels = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
```

**同理**Critic 也输入的是：

- `seqs`：整条序列（prompt + answer）；`attention_mask`：哪些 token 有效；`num_actions`：回答部分 token 数（通常用来只取 answer 那段的 value）

输出价值 value（方法：把每个位置的 hidden state 过一个 **线性层 → 标量**）

总结：**Actor（策略模型）最后那一层 “Language Modeling Head（LM head）” 来输出“下一个 token 的分布”。**

**Critic模型在最后一层加 Value 头输出V(s)。**

[案例](https://www.notion.so/2b176eb9082280efbc2bdf6e3f5a6456?pvs=21)

1. **用 PPO 更新 Actor（策略）和 Critic（价值网络）**

根据：新 logprob；旧 logprob；advantage A_t；action_mask

计算 **PPO policy loss（clip ratio loss）**：

$loss
=
-\mathbb{E}_t \big[
\min(
r_t(\theta) A_t,\ 
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\, A_t
)
\big]$

```
L_policy = - min(r_t*A_t, clip(r_t)*A_t)

# logprobs_new: 当前模型对动作的 logπ_θ
# logprobs_old: 采样时的 logπ_old（detach 存下来）
# advantages:   A_t
ratio = torch.exp(logprobs_new - logprobs_old)
# 1. unclipped
pg_unclipped = ratio * advantages
# 2. clipped
ratio_clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
pg_clipped = ratio_clipped * advantages
# 3. 取 min，并取负号变成 loss
policy_loss = -torch.mean(torch.min(pg_unclipped, pg_clipped))
```

反向传播更新策略模型 πθ。

---

② Critic（价值）更新

根据：新 value；return

计算 **MSE value loss**：

```
L_value = (V(s_t) - return)^2

```

更新 critic 模型。

1. **新策略继续 rollout → 新 experience → 继续训练**

整体伪代码：LLM + PPO（简化版）

```python
# 已有:
# policy_model = LLM with value head (π_θ)
# ref_model    = frozen LLM (π_ref)
# reward_model = trained RM (R_ϕ)

for step in range(num_rl_steps):
    # === 1. 收集 rollouts ===
    prompts = sample_prompts(batch_size)

    with torch.no_grad():
        # 用当前策略生成回答，记录 logprob 和 value
        trajectories = policy_model.generate(
            prompts,
            return_logprobs=True,
            return_values=True,
        )

        # 参考模型 logprob
        ref_logprobs = ref_model.get_logprobs(trajectories)

        # 奖励模型打分
        rm_scores = reward_model.get_reward(prompts, trajectories)

        # 计算 per-token KL 惩罚 + 最终 reward
        rewards = compute_rewards(
            rm_scores,
            trajectories.logprobs,
            ref_logprobs,
            beta=kl_coef,
        )

        # 用 GAE 算 advantage 和 returns
        advantages, returns = compute_gae(
            rewards,
            trajectories.values,
            gamma=gamma,
            lam=gae_lambda,
        )

    # 把 rollout 切成小 batch，做 K 轮 PPO 更新
    for epoch in range(ppo_epochs):
        for mini_batch in make_minibatches(trajectories, advantages, returns):
            logprobs_new, values_new = policy_model(
                mini_batch["input_ids"],
                compute_logprobs=True,
                compute_values=True,
            )

            loss = ppo_loss(
                logprobs_new,
                values_new,
                mini_batch["old_logprobs"],
                mini_batch["old_values"],
                mini_batch["advantages"],
                mini_batch["returns"],
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

```