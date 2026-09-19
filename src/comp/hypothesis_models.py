"""Experimental BS pair encoder; not wired into product inference (SPEC-32)."""
import torch
from torch import nn
from scripts.train_unet import UNet, block


class SiameseUNet(nn.Module):
    def __init__(self,width=32,depth=7):
        super().__init__()
        widths=[width*2**i for i in range(depth)]
        self.down=nn.ModuleList([block(9 if i==0 else widths[i-1],w) for i,w in enumerate(widths)])
        self.pool=nn.MaxPool2d(2)
        self.fuse=nn.ModuleList([nn.Conv2d(3*w+(11 if i==0 else 0),w,1) for i,w in enumerate(widths)])
        self.up=nn.ModuleList([nn.ConvTranspose2d(widths[i],widths[i-1],2,2) for i in range(depth-1,0,-1)])
        self.conv=nn.ModuleList([block(widths[i-1]*2,widths[i-1]) for i in range(depth-1,0,-1)])
        self.head=nn.Conv2d(width,4,1)

    def forward(self,x):
        a,b=x[:,:9],x[:,9:18]; skips=[]
        for i,(encoder,fuse) in enumerate(zip(self.down,self.fuse)):
            # One joint batch gives identical BN treatment to both dates.
            n=len(a); both=encoder(torch.cat([a,b],0)); a,b=both[:n],both[n:]
            features=[a,b,b-a]
            if i==0: features.append(x[:,18:])
            skips.append(fuse(torch.cat(features,1)))
            if i<len(self.down)-1: a,b=self.pool(a),self.pool(b)
        z=skips[-1]
        for i,(up,conv) in enumerate(zip(self.up,self.conv)): z=conv(torch.cat([up(z),skips[-2-i]],1))
        return self.head(z)


def make_model(variant,width=32,depth=7):
    if variant=='siam': return SiameseUNet(width,depth)
    return UNet(11 if variant=='optical' else 29,classes=4,w=width,depth=depth)
