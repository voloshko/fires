"""SPEC-92: reproducible synthetic CPU check of the official competitor loader."""
import argparse
import json
from pathlib import Path
import tempfile
import torch
import yaml
from terratorch.cli_tools import LightningInferenceModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--size',type=int,default=32)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('immutable preflight result already exists')
    cfg=yaml.safe_load((a.model_root/'burn_scars_config.yaml').read_text())
    cfg['model']['init_args']['model_args']['backbone_pretrained']=False
    cfg['trainer'].update(logger=False,callbacks=[],enable_checkpointing=False,
                          accelerator='cpu',devices=1,precision='32-true')
    with tempfile.TemporaryDirectory() as d:
        config=Path(d)/'config.yaml';config.write_text(yaml.safe_dump(cfg))
        net=LightningInferenceModel.from_config(str(config),str(a.model_root/'Prithvi_EO_V2_300M_BurnScars.pt')).model.cpu().eval()
        with torch.no_grad():
            result=net(torch.zeros(1,6,a.size,a.size));logits=getattr(result,'output',result)
        report=dict(evidence_class='SYNTHETIC',shape=list(logits.shape),finite=bool(torch.isfinite(logits).all()),
                    non_claims=['Does not measure accuracy or real-data inference performance.'])
        if report['shape'] != [1,2,a.size,a.size] or not report['finite']:raise ValueError('competitor preflight failed')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
