from torchvision.transforms import GaussianBlur,Normalize,Resize
import torchvision.models as models
import torch

NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

class Seg_CESAR(torch.nn.Module):
    
    def __init__(self, SEG_NET,CUDA = True):
        super(Seg_CESAR, self).__init__()

        self.segnet = SEG_NET
        self.mid_conv = torch.nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1,bias=True)
        
        ## Frezze vessel extraction
        for param in self.segnet.parameters():
            param.requires_grad = False
        
        if CUDA:
            self.densenet = models.densenet161(pretrained = True).cuda(0)
        else:
            self.densenet = models.densenet161(pretrained = True)
        self.densenet.classifier = torch.nn.Linear(in_features=2208,out_features=2,bias = True)
        
        
    def forward(self, x):
        # Vessels extraction
        
        
        x = self.segnet(x)
        
        x = GaussianBlur(3)(x)
        
        x = torch.cat((x,x,x),1)
        ## Normalization for using pre-trained network
        x = Normalize(NORMALIZATION[0],NORMALIZATION[1])(x)
        x = Resize(224)(x)
        
        # Decoder
        
        x = self.densenet(x)
        return x        

    
    def load_from_filename(self, model_path):
        """Load weights from filename."""
        th = torch.load(model_path)  # load the weigths
        self.load_state_dict(th)