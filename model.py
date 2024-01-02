import torch
from torchvision.models import efficientnet_v2_s,EfficientNet_V2_S_Weights
from torch import nn
def model(classes):
    model=efficientnet_v2_s()
    device="cuda" if torch.cuda.is_available() else "Cpu"
    for param in model.features.parameters():
        param=False
    model.classifier=nn.Sequential(
        nn.Dropout(p=0.2,inplace=True),
        nn.Linear(in_features=1280,out_features=classes,bias=True)
    )


    
    weights=EfficientNet_V2_S_Weights.DEFAULT
    transforms=weights.transforms()
    return model,transforms
def main():
    pass
if __name__ =="__main__":
    main()

    

