import json
import pytest
import torch
from src.comp.accuracy_artifacts import save_checkpoint
from src.comp.accuracy_contract import sha256_file
from scripts.score_russian_transfer import model_recipe_audit


def test_transfer_checks_actual_fit_population_and_initializers(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    initializers=[]
    for i in (1,2,3):
        p=tmp_path/f'init-{i}.pt';p.write_bytes(str(i).encode());initializers.append(p)
    models={v:[] for v in ['control','reserve']}
    for v in models:
        for seed in (1,2,3):
            d=tmp_path/f'{v}-{seed}';d.mkdir();p=d/'model.pt'
            meta=dict(source_hashes={'train':'hash'},software={'torch':'version'},base_manifest='base',
                      russian_manifest='russia',width=32,depth=7,epochs=30,batch=8,steps_per_epoch=58,
                      classes=2,variant=v,seed=seed,smoke=False,
                      initializer_recipe_id='recipe',initializer_result_sha256='result',
                      fit_ids=['base']+(['russia'] if v=='reserve' else []),
                      initialization_sha256=sha256_file(initializers[seed-1]))
            save_checkpoint(p,{'w':torch.ones(1)},meta);(d/'data_manifest.json').write_text(json.dumps(meta));models[v].append(p)
    model_recipe_audit(models,'base','russia',initializers,['base'],['russia'])
    with pytest.raises(ValueError):model_recipe_audit(models,'base','russia',initializers,['base'],['russia','evaluation'])
