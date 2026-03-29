import requests
import json
import threading
import time
from typing import Dict, Any

BASE_URL = "http://localhost:7860"

def print_result(name: str, passed: bool, details: str = ""):
    icon = "✅" if passed else "❌"
    print(f"{icon} {name}")
    if details:
        print(f"   {details}")

def run_stress_test():
    print("="*50)
    print("  LEXCRISIS ADVERSARIAL STRESS TEST SUITE")
    print("="*50)

    # Health check
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except:
        print("ERROR: Environment server is not running on port 7860.")
        return

    # TEST 1: Malformed JSON payload
    try:
        resp = requests.post(f"{BASE_URL}/step", data="INVALID JSON", headers={"Content-Type": "application/json"})
        # FastAPI should catch this and return 422 Unprocessable Entity
        print_result("Malformed JSON Rejection", resp.status_code == 422, f"Expected 422, got {resp.status_code}")
    except Exception as e:
        print_result("Malformed JSON Rejection", False, str(e))

    # TEST 2: Missing Required Fields (action_type)
    try:
        resp = requests.post(f"{BASE_URL}/step", json={"parameters": {}})
        print_result("Missing Field Validation (action_type)", resp.status_code == 422, f"Expected 422, got {resp.status_code}")
    except Exception as e:
        print_result("Missing Field Validation", False, str(e))

    # TEST 3: Invalid action type for current task
    try:
        requests.post(f"{BASE_URL}/reset", json={"task_id": "task_1"})
        resp = requests.post(f"{BASE_URL}/step", json={"action_type": "classify_privilege", "parameters": {}})
        data = resp.json()
        expected = "Invalid action" in data.get("observation", {}).get("feedback", "")
        print_result("Action Scope Enforced (T1 cannot run T2 action)", expected, data.get("observation", {}).get("feedback", "")[:80])
    except Exception as e:
        print_result("Action Scope Enforced", False, str(e))

    # TEST 4: Type confusion (sending int where string is expected)
    try:
        requests.post(f"{BASE_URL}/reset", json={"task_id": "task_1"})
        resp = requests.post(f"{BASE_URL}/step", json={"action_type": "review_client", "parameters": {"client_id": 12345}})
        data = resp.json()
        expected = "Unknown client" in data.get("observation", {}).get("feedback", "")
        print_result("Type Confusion Handled (Int instead of Str)", expected, data.get("observation", {}).get("feedback", "")[:80])
    except Exception as e:
        print_result("Type Confusion Handled", False, str(e))

    # TEST 5: Out of bounds state corruption attempts
    try:
        requests.post(f"{BASE_URL}/reset", json={"task_id": "task_1"})
        resp = requests.post(f"{BASE_URL}/step", json={"action_type": "check_conflict", "parameters": {"client_a": "A"*10000, "client_b": "B"*10000}})
        data = resp.json()
        expected_status = resp.status_code == 200
        expected_feedback = "No conflict found" in data.get("observation", {}).get("feedback", "")
        print_result("Buffer Overflow/Massive String Handled", expected_status and expected_feedback, "Successfully rejected massive strings gracefully")
    except Exception as e:
        print_result("Buffer Overflow Handled", False, str(e))

    # TEST 6: Privilege Waiver Trap Resilience
    try:
        requests.post(f"{BASE_URL}/reset", json={"task_id": "task_3"})
        # Step into the trap
        resp = requests.post(f"{BASE_URL}/step", json={
            "action_type": "respond_discovery", 
            "parameters": {"request_id": "14", "response_type": "produce", "objections": ""}
        })
        data = resp.json()
        reward = data.get("reward", 0)
        feedback = data.get("observation", {}).get("feedback", "")
        alerts = data.get("observation", {}).get("ethical_alerts", [])
        
        trap_sprung = (reward < 0) and ("WAIVER" in feedback or "PRIVILEGE" in feedback) and (len(alerts) > 0)
        print_result("Privilege Waiver Trap Triggered", trap_sprung, f"Reward: {reward}, Alerts: {len(alerts)}")
    except Exception as e:
        print_result("Privilege Waiver Trap", False, str(e))

    print("\nAdversarial Stress Test Complete.")

if __name__ == "__main__":
    run_stress_test()
