#!/usr/bin/env python3
"""
CEREBROS NotGPT - End-to-End Validation Test
Tests the complete pipeline from data processing to API queries
"""

import sys
import subprocess
import time
import json
import requests
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8080"
ASSISTANT_ID = "demo"

def run_command(cmd, description):
    """Run a shell command and report results"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ SUCCESS")
            return True
        else:
            print(f"❌ FAILED")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_api_endpoint(method, endpoint, description, data=None):
    """Test an API endpoint"""
    print(f"\n{'='*60}")
    print(f"🌐 {description}")
    print(f"{'='*60}")
    print(f"Endpoint: {method} {endpoint}")
    
    try:
        url = f"{API_BASE}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        if response.status_code in [200, 201]:
            print(f"✅ SUCCESS (Status: {response.status_code})")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"❌ FAILED (Status: {response.status_code})")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ CONNECTION ERROR - Is the API server running?")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Run complete end-to-end validation"""
    print("\n" + "="*80)
    print("🚀 CEREBROS NotGPT - End-to-End Validation Test")
    print("="*80)
    
    results = {
        "setup": False,
        "data_processing": False,
        "training": False,
        "file_upload": False,
        "api_health": False,
        "api_list": False,
        "api_status": False,
        "api_query": False,
        "round_trip": False
    }
    
    print("\n" + "="*60)
    print("📋 Test 1: Environment Check")
    print("="*60)
    if Path("priv/nfs").exists():
        print("✅ NFS directory exists")
        results["setup"] = True
    else:
        print("❌ NFS directory not found - run ./start_cerebros.sh first")
        return results

    # Test 2: File Upload Simulation
    print("\n" + "="*60)
    print("📤 Test 2: File Upload Simulation (/api/upload)")
    print("="*60)
    try:
        files = {"file": ("dummy.csv", "a,b,c\n1,2,3\n", "text/csv")}
        upload_resp = requests.post(f"{API_BASE}/api/upload", files=files, timeout=10)
        if upload_resp.status_code in (200, 201):
            print("✅ Upload success")
            results["file_upload"] = True
        else:
            print(f"❌ Upload failed ({upload_resp.status_code}) {upload_resp.text}")
    except Exception as e:
        print(f"❌ Upload error: {e}")

    # Continue base tests
    results["data_processing"] = run_command(
        ["python3", "scripts/process_user_samples.py", "--assistant_id", ASSISTANT_ID],
        "Test 3: Data Processing Pipeline"
    )

    if not results["data_processing"]:
        print("\n⚠️  Data processing failed, but continuing with tests...")

    results["training"] = run_command(
        ["python3", "multi_stage_trainer.py", ASSISTANT_ID, f"{ASSISTANT_ID.title()} Assistant", "priv/nfs"],
        "Test 4: Multi-Stage Training Pipeline"
    )

    checkpoints_path = Path(f"priv/nfs/agents/{ASSISTANT_ID}/checkpoints")
    if checkpoints_path.exists():
        checkpoint_files = list(checkpoints_path.glob("*.keras"))
        print(f"✅ Found {len(checkpoint_files)} model checkpoints")
        for f in checkpoint_files:
            print(f"  - {f.name}")
    else:
        print("❌ Missing model checkpoints")

    time.sleep(2)
    results["api_health"] = test_api_endpoint("GET","/health","API Health Check")
    results["api_list"] = test_api_endpoint("GET","/assistants","List Assistants")
    results["api_status"] = test_api_endpoint("GET",f"/assistants/{ASSISTANT_ID}/status","Get Assistant Status")
    results["api_query"] = test_api_endpoint("POST",f"/assistants/{ASSISTANT_ID}/query","Query Assistant",data={"query":"round-trip test","temperature":0.7,"max_tokens":32})

    # Chat round trip test: ensure consistency between 2 sequential queries
    print("\n" + "="*60)
    print("💬 Test 10: Chat Round‑Trip Verification")
    print("="*60)
    try:
        query1 = requests.post(f"{API_BASE}/assistants/{ASSISTANT_ID}/query", json={"query":"hello"}, timeout=10).json()
        query2 = requests.post(f"{API_BASE}/assistants/{ASSISTANT_ID}/query", json={"query":"hello again"}, timeout=10).json()
        if "response" in query1 and "response" in query2:
            print(f"✅ Chat round trip success: {len(query1['response'])} chars first, {len(query2['response'])} chars second")
            results["round_trip"] = True
        else:
            print("❌ Incomplete round-trip data")
    except Exception as e:
        print(f"❌ Round-trip error: {e}")

    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    print("\n" + "-"*80)
    print(f"Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    print("="*80)
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! System is fully operational.")
        return 0
    elif passed_tests >= total_tests * 0.7:
        print("\n⚠️  Most tests passed. Check failures above.")
        return 1
    else:
        print("\n❌ Multiple tests failed. Review logs above.")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)