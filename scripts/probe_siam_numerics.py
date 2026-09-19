"""Replay the saved failed batch, keeping all learned weights for a range probe."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.hypothesis_models import make_model
from src.comp.hypothesis_lab import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    b=torch.load(Path(args.run)/'debug_state.pt',map_location='cuda',weights_only=False)
    old=make_model(b['variant'],b['width'],b['depth'],normalize_fusion=False).cuda().train();old.load_state_dict(b['state']);captured={}
    class Captured(Exception):pass
    def capture(module,inputs):captured['x']=inputs[0].detach();raise Captured()
    hook=old.conv[0][0].register_forward_pre_hook(capture)
    with torch.no_grad():
        try:
            with torch.autocast('cuda'):old(b['x'].cuda())
        except Captured:pass
        hook.remove();x=captured['x'];layer=old.conv[0][0]
        with torch.autocast('cuda'):half=layer(x)
        with torch.autocast('cuda',enabled=False):full=layer(x.float())
        report=dict(original_fp16_nonfinite=int((~torch.isfinite(half)).sum()),original_fp32_finite=bool(torch.isfinite(full).all()),original_fp32_absmax=float(full.abs().max()),fp16_limit=float(torch.finfo(torch.float16).max))
        fixed=make_model(b['variant'],b['width'],b['depth'],normalize_fusion=True).cuda().train()
        translated={}
        for k,v in b['state'].items():
            if k.startswith('fuse.'):
                parts=k.split('.'); k='.'.join(parts[:2]+['0']+parts[2:])
            translated[k]=v
        missing,unexpected=fixed.load_state_dict(translated,strict=False)
        assert not unexpected and all(k.startswith('fuse.') and '.1.' in k for k in missing)
        before={}
        def capture_fixed(module,inputs):before['input_max']=float(inputs[0].abs().max())
        h=fixed.conv[0][0].register_forward_pre_hook(capture_fixed)
        with torch.autocast('cuda'):out=fixed(b['x'].cuda())
        h.remove();report.update(fixed_fp16_finite=bool(torch.isfinite(out).all()),fixed_decoder_input_max=before['input_max'],non_claim='Numerical replay only, no accuracy measurement; real trials restart from scratch.')
    write_json(args.out,report);print(json.dumps(report,indent=2))
    if not report['fixed_fp16_finite']:raise RuntimeError('numerical repair failed')
if __name__=='__main__':main()
