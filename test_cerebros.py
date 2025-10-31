#!/usr/bin/env python3
"""
Quick Test of Cerebros Multi-Stage Training System
"""

import subprocess
import time
import sys

def test_training_pipeline():
    """Test the training pipeline directly"""
    print("=" * 60)
    print("Testing Cerebros Multi-Stage Training Pipeline")
    print("=" * 60)
    
    agent_id = "test-demo-001"
    agent_name = "Demo Assistant"
    
    print(f"\n📝 Testing with Agent ID: {agent_id}")
    print(f"   Agent Name: {agent_name}\n")
    
    result = subprocess.run(
        [
            "python3",
            "cerebros-core-algorithm-alpha/multi_stage_trainer.py",
            agent_id,
            agent_name
        ],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    if result.returncode == 0:
        print("\n✅ Training pipeline completed successfully!")
        print(f"\n📂 Check results at: priv/nfs/agents/{agent_id}/")
        return True
    else:
        print("\n❌ Training pipeline failed!")
        print(f"Error: {result.stderr}")
        return False

if __name__ == "__main__":
    success = test_training_pipeline()
    sys.exit(0 if success else 1)
