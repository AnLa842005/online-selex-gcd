"""Report the actual runtime; run with the intended environment's Python."""
import json
import platform
import torch
import torchvision
import timm
import sklearn
import kmeans_pytorch


def main():
    print(json.dumps(dict(python=platform.python_version(), torch=torch.__version__,
                         torchvision=torchvision.__version__, timm=timm.__version__,
                         sklearn=sklearn.__version__, cuda=torch.version.cuda,
                         cuda_available=torch.cuda.is_available(),
                         gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                         kmeans_module=kmeans_pytorch.__file__), indent=2))
    x = torch.ones(2, device='cuda' if torch.cuda.is_available() else 'cpu')
    assert (x + x).sum().item() == 4


if __name__ == '__main__':
    main()
