import torch
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))
    
    # Thử thực hiện một số phép tính trên CUDA
    x = torch.rand(5, 3).cuda()
    print("Tensor on CUDA:", x.device)
    print("Tensor sample:", x[:2])
else:
    print("CUDA không được phát hiện. Kiểm tra driver và cài đặt CUDA.")
    print("Thông tin hệ thống:")
    import platform
    print("OS:", platform.platform())
    
    # Thử import các thư viện liên quan đến CUDA
    try:
        import nvidia.cublas
        print("nvidia.cublas available")
    except ImportError:
        print("nvidia.cublas not available")
    
    try:
        import torch._C
        print("torch._C available")
    except ImportError:
        print("torch._C not available") 