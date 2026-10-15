import torch
import time

def run_burn_in(duration_seconds=300):
    """
    Runs a CUDA burn-in test for a specified duration.
    """
    # 1. Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU (not a stress test).")
        return

    device = torch.device("cuda")
    print(f"Starting burn-in on: {torch.cuda.get_device_name(0)}")

    # 2. Setup large matrices to maximize tensor core usage
    # Increase size to fill more memory if needed
    size = 10000 
    matrix_a = torch.randn(size, size, device=device, requires_grad=True)
    matrix_b = torch.randn(size, size, device=device, requires_grad=True)

    end_time = time.time() + duration_seconds
    iterations = 0
    
    # 3. Continuous calculation loop
    try:
        while time.time() < end_time:
            # Perform matrix multiplication (high GPU load)
            result = torch.matmul(matrix_a, matrix_b)
            
            # Perform a backward pass to stress memory bandwidth
            result.sum().backward()
            
            # Synchronize to ensure tasks are completed
            torch.cuda.synchronize()
            
            iterations += 1
            if iterations % 10 == 0:
                print(f"Iterations completed: {iterations}, Time remaining: {int(end_time - time.time())}s", end='\r')
                
    except KeyboardInterrupt:
        print("\nBurn-in stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nBurn-in finished.")

if __name__ == "__main__":
    # Run for 5 minutes (300 seconds)
    run_burn_in(60)
