"""Experimental BS pair encoder; not wired into product inference (SPEC-32)."""
import torch
from torch import nn
from scripts.train_unet import UNet, block


class SiameseUNet(nn.Module):
    def __init__(self,width=32,depth=7,normalize_fusion=True,fusion='full',aux_channels=11):
        super().__init__()
        # SPEC-51: fusion='diff' — в декодер идёт только разность b−a (FC-Siam-diff),
        # без внешнего вида каждой даты по отдельности; 'full' — [a, b, b−a], как раньше.
        if fusion not in ('full','diff'): raise ValueError(fusion)
        self.fusion=fusion; k=3 if fusion=='full' else 1; self.aux_channels=aux_channels
        widths=[width*2**i for i in range(depth)]
        self.down=nn.ModuleList([block(9 if i==0 else widths[i-1],w) for i,w in enumerate(widths)])
        self.pool=nn.MaxPool2d(2)
        self.fuse=nn.ModuleList([
            nn.Sequential(nn.Conv2d(k*w+(aux_channels if i==0 else 0),w,1),nn.BatchNorm2d(w))
            if normalize_fusion else nn.Conv2d(k*w+(aux_channels if i==0 else 0),w,1)
            for i,w in enumerate(widths)])
        self.up=nn.ModuleList([nn.ConvTranspose2d(widths[i],widths[i-1],2,2) for i in range(depth-1,0,-1)])
        self.conv=nn.ModuleList([block(widths[i-1]*2,widths[i-1]) for i in range(depth-1,0,-1)])
        self.head=nn.Conv2d(width,4,1)

    def forward(self,x):
        a,b=x[:,:9],x[:,9:18]; skips=[]
        for i,(encoder,fuse) in enumerate(zip(self.down,self.fuse)):
            # One joint batch gives identical BN treatment to both dates.
            n=len(a); both=encoder(torch.cat([a,b],0)); a,b=both[:n],both[n:]
            features=[a,b,b-a] if self.fusion=='full' else [b-a]
            if i==0: features.append(x[:,18:])
            skips.append(fuse(torch.cat(features,1)))
            if i<len(self.down)-1: a,b=self.pool(a),self.pool(b)
        z=skips[-1]
        for i,(up,conv) in enumerate(zip(self.up,self.conv)): z=conv(torch.cat([up(z),skips[-2-i]],1))
        return self.head(z)


def make_model(variant,width=32,depth=7,normalize_fusion=True,in_channels=None,fusion='full'):
    # SPEC-57: у сиама первые 18 каналов — пары полос, остальное — вспомогательные (11 оптических + хуки).
    if variant=='siam': return SiameseUNet(width,depth,normalize_fusion,fusion,aux_channels=(in_channels-18) if in_channels else 11)
    # SPEC-47: число каналов берётся из данных (FEATURES=swir даёт 17 вместо 11).
    return UNet(in_channels or (11 if variant=='optical' else 29),classes=4,w=width,depth=depth)


def masked_binary_loss(logits,target):
    """Ignore padded pixels in BOTH loss terms (external CEMS source geometry)."""
    import torch.nn.functional as F
    known=target!=255
    if not known.any(): raise ValueError('empty supervision')
    p=logits.float().softmax(1)[:,1]*known
    t=(target==1).float()
    return F.cross_entropy(logits,target,ignore_index=255)+1-(2*(p*t).sum()+1)/(p.sum()+t.sum()+1)


def two_stage_loss(logits,y,weights=None):
    """SPEC-51: двухэтапная потеря в одной сети. Этап 1 — гарь/фон: бинарная CE по
    логиту logsumexp(классы 1..3) − логит(0) плюс dice; этап 2 — степень только на
    пикселях истинной гари (CE по классам 1..3). Фон не тянет степень, степень не
    тянет фон — в отличие от общей 4-классовой CE."""
    import torch.nn.functional as F
    burn_logit=torch.logsumexp(logits[:,1:],1)-logits[:,0]; truth=(y>0).float()
    p=torch.sigmoid(burn_logit)
    loss=F.binary_cross_entropy_with_logits(burn_logit,truth)+1-(2*(p*truth).sum()+1)/(p.sum()+truth.sum()+1)
    if truth.any():
        sev=logits[:,1:].permute(0,2,3,1)[y>0]; loss=loss+F.cross_entropy(sev,(y[y>0]-1))
    return loss


def soft_edge_targets(y,k=3,classes=4):
    """SPEC-52: мягкие метки у кромки — one-hot, усреднённый окном k×k. Вдали от
    границ классов совпадает с one-hot; в кольце шириной k//2 по обе стороны
    кромки масса делится между соседними классами пропорционально их доле в окне."""
    import torch.nn.functional as F
    onehot=F.one_hot(y.clamp(0,classes-1),classes).permute(0,3,1,2).float()
    return F.avg_pool2d(onehot,k,stride=1,padding=k//2,count_include_pad=False)


def soft_edge_loss(logits,y,k=3,weights=None):
    """CE по мягким меткам (вес класса — по жёсткой метке пикселя) + dice гарь/фон, как в базовой потере."""
    soft=soft_edge_targets(y,k,logits.shape[1]); logp=logits.float().log_softmax(1)
    ce=-(soft*logp).sum(1)
    if weights is not None: ce=ce*weights[y]
    p=1-logp[:,0].exp(); truth=(y>0).float()
    return ce.mean()+1-(2*(p*truth).sum()+1)/(p.sum()+truth.sum()+1)
