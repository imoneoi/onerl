import torch
import time

def test_to_device(x, device, test_count):
    start = time.time()
    for _ in range(test_count):
        x.to(device)
        torch.cuda.synchronize(device)
    return time.time() - start

def test_to_device_with_pin(x, device, test_count):
    start = time.time()

    xpin = torch.zeros_like(x)
    with torch.cuda.device(device):
        torch.cuda.cudart().cudaHostRegister(xpin.data_ptr(), xpin.numel() * xpin.element_size(), 0)

    for _ in range(test_count):
        xpin.copy_(x)
        xpin.to(device)
        torch.cuda.synchronize(device)

    with torch.cuda.device(device):
        torch.cuda.cudart().cudaHostUnregister(xpin.data_ptr())

    return time.time() - start

def test_from_device(x, device, test_count):
    start = time.time()
    xdev = torch.zeros(x.shape, dtype=x.dtype, device=device)
    for _ in range(test_count):
        x.copy_(xdev, non_blocking=True)
        torch.cuda.synchronize(device)
    return time.time() - start


# 16MB
size = 16 * 1024 * 1024
test_count = 100

x = torch.zeros(size, dtype=torch.uint8)
# pin to dev 0

with torch.cuda.device("cuda:1"):
    torch.cuda.cudart().cudaHostRegister(x.data_ptr(), x.numel() * x.element_size(), 0)


print("To device 0", test_to_device(x, "cuda:0", test_count))
print("To device 1", test_to_device(x, "cuda:1", test_count))

print("To device 0 with pin", test_to_device_with_pin(x, "cuda:0", test_count))
print("To device 1 with pin", test_to_device_with_pin(x, "cuda:1", test_count))

print("From device 0", test_from_device(x, "cuda:0", test_count))
print("From device 1", test_from_device(x, "cuda:1", test_count))
