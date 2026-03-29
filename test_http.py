"""Quick HTTP endpoint smoke test."""
import requests
import json

base = "http://localhost:7860"

# Health
r = requests.get(f"{base}/health")
print("Health:", r.json())

# Tasks
r = requests.get(f"{base}/tasks")
for tid, info in r.json().items():
    print(f"  {tid}: {info['name']} ({info['difficulty']})")

# Reset task_1
r = requests.post(f"{base}/reset", json={"task_id": "task_1"})
data = r.json()
print("\nReset task_1 OK, docs:", len(data["observation"]["documents"]))

# Step
r = requests.post(f"{base}/step", json={"action_type": "review_client", "parameters": {"client_id": "CLIENT-001"}})
data = r.json()
print("Step reward:", data["reward"], "done:", data["done"])
print("Content:", "yes" if data["observation"]["current_content"] else "no")

# State
r = requests.get(f"{base}/state")
data = r.json()
print("State - step:", data["step_count"], "score:", data["score"])

# Invalid action
r = requests.post(f"{base}/step", json={"action_type": "invalid_garbage", "parameters": {}})
data = r.json()
print("Invalid action reward:", data["reward"])

# Reset task_3
r = requests.post(f"{base}/reset", json={"task_id": "task_3"})
data = r.json()
obs = data["observation"]
print("\nReset task_3 OK")
print("  Documents:", len(obs["documents"]))
print("  Deadlines:", len(obs.get("active_deadlines", [])))
print("  Actions:", obs["available_actions"])

# Adversarial test: malformed params
r = requests.post(f"{base}/step", json={"action_type": "review_event", "parameters": {"event_id": "BOGUS"}})
data = r.json()
print("Bogus event reward:", data["reward"])

print("\nALL HTTP TESTS PASSED")
