import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Load results
with open("results.json") as f:
    data = json.load(f)

# Group by task
task_rewards = defaultdict(list)

for d in data:
    task_rewards[d["task"]].append(d["reward"])

# Average reward per task
tasks = []
avg_rewards = []

for task, rewards in task_rewards.items():
    tasks.append(task)
    avg_rewards.append(sum(rewards) / len(rewards))

# Plot
plt.figure()
plt.bar(tasks, avg_rewards)
plt.xlabel("Tasks")
plt.ylabel("Average Reward")
plt.title("Model Performance per Task")
plt.show()