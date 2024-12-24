import torch
from descritization import conversion
import sys
from plot_light_config import *
import numpy as np


if __name__ == '__main__':
    model = sys.argv[1]
    load_dict = torch.load(model)
    print('Best configuration given by trained model in x,y,z format')
    xyzs = []
    for i in [torch.argmax(i).item() for i in load_dict['state_dict']['weight']]:
        print(conversion.get_config(i))
        xyzs.append(conversion.get_config(i))
    plot_lighting(np.array(xyzs), './')
    print('Light Configs plot saved as lighting.png')
        
    