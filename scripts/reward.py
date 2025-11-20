# simple_reward.py
import tempfile
import subprocess
from pathlib import Path

def get_dafny_reward(code: str, dafny_path: str = "dafny", timeout: int = 30) -> int:
    """
    Simple Dafny reward function. Returns 1 if code verifies, 0 otherwise.
    
    Args:
        code: Dafny code string to verify
        dafny_path: Path to dafny executable
        timeout: Verification timeout in seconds
        
    Returns:
        int: 1 if verification successful, 0 otherwise
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dfy', delete=False) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        # Run Dafny verification
        cmd = [dafny_path, "verify", temp_path, "--verification-time-limit", str(timeout)]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5
        )
        
        # Check output for success
        output = result.stdout + result.stderr
        if "0 errors" in output and "verified" in output.lower():
            return 1
        else:
            return 0
            
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return 0
    finally:
        # Clean up
        Path(temp_path).unlink(missing_ok=True)

# Example usage
if __name__ == "__main__":
    test_code = """
    method Max(a: int, b: int) returns (max: int)
        ensures max >= a && max >= b
    {
        if a > b {
            max := a;
        } else {
            max := b;
        }
    }
    """
    
    reward = get_dafny_reward(test_code)
    print(f"Reward: {reward}")  # Should print 1 for correct code