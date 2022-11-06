import os
import torch

if __name__ == '__main__':
    # print(os.listdir())
    model_path = '/home/oscar/Desktop/Research_support/HandMesh/checkpoint_last.pt'
    to_path    = '/home/oscar/Desktop/Research_support/HandMesh/output.pth'
    checkpoint = torch.load(to_path)
    print(checkpoint.keys())
    weight = checkpoint['model_state_dict']
    # torch.save(weight, to_path)
