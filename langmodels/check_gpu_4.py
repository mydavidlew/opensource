import sys
import time
import torch

def inspect_cuda_environment():
    print("=" * 60)
    print("CUDA & GPU COMPUTE ENVIRONMENT CHECK")
    print("=" * 60)
    
    # 1. System & PyTorch Baseline
    print(f"[*] Python Version       : {sys.version.split()[0]}")
    print(f"[*] PyTorch Version      : {torch.__version__}")
    
    # 2. CUDA Availability
    cuda_available = torch.cuda.is_available()
    print(f"[*] CUDA Supported Build : {torch.version.cuda}")
    print(f"[*] CUDA Driver/Device   : {'AVAILABLE' if cuda_available else 'UNAVAILABLE'}")
    
    if not cuda_available:
        print("\n[!] Result: System is relying strictly on CPU execution.")
        print("    If a physical GPU is present, verify driver installation and toolkit dependencies.")
        return

    # 3. Target GPU Hardware Telemetry
    gpu_count = torch.cuda.device_count()
    print(f"CUDA Available: Yes, found {gpu_count} GPU(s).")
    for i in range(gpu_count):
        current_device = i # torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        capability = torch.cuda.get_device_capability(current_device)
        properties = torch.cuda.get_device_properties(current_device)
        vram_bytes = torch.cuda.get_device_properties(current_device).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        print(f"\n--- Device [{current_device}] Details ---")
        print(f"[*] Total GPUs Detected    : {gpu_count}")
        print(f"[*] Primary Target GPU     : [{current_device}] {device_name}")
        print(f"[*] Compute Capability     : sm_{capability[0]}{capability[1]}")
        print(f"[*] CUDA Capability        : {properties.major}.{properties.minor}")
        print(f"[*] Dedicated VRAM         : {vram_gb:.2f} GB")
        #print(f"[*] Total Memory           : {properties.total_memory/(1024**3):.2f} GB")
        print(f"[*] Memory Usage Allocated : {round(torch.cuda.memory_allocated(current_device)/1024**3,1)} GB")
        print(f"[*] Memory Cached          : {round(torch.cuda.memory_reserved(current_device)/1024**3,1)} GB")
        print("-" * 60)

        # 4. Compute Benchmark Execution (Tensor Multiplication)
        print("[*] Executing Matrix Compute Benchmark (FP32)...")
        matrix_dim = 8192  # 8192 x 8192 dense matrix
    
        try:
            # Initialize random tensors directly on the GPU device
            device = torch.device(f"cuda:{current_device}")
        
            # Warmup execution
            a = torch.randn(matrix_dim, matrix_dim, device=device, dtype=torch.float32)
            b = torch.randn(matrix_dim, matrix_dim, device=device, dtype=torch.float32)
            c = torch.matmul(a, b)
            torch.cuda.synchronize()

            # Timed execution run
            start_time = time.perf_counter()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()  # Enforce synchronous blocking to capture exact compute time
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            print(f"[✓] Execution Successful!")
            print(f"    - Matrix Dimension   : {matrix_dim} x {matrix_dim}")
            print(f"    - Execution Latency  : {elapsed_ms:.2f} ms")
            print(f"    - VRAM Allocation    : {torch.cuda.memory_allocated(device) / (1024**2):.2f} MB")
        
        except Exception as e:
            print(f"[!] Execution Failed during tensor processing: {e}")

    print("=" * 60)

if __name__ == "__main__":
    inspect_cuda_environment()
