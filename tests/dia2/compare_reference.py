#!/usr/bin/env python3
"""Compare deterministic native temporal/depformer/codec outputs with official PyTorch."""
import argparse, json, sys, types
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
ap=argparse.ArgumentParser();ap.add_argument('package',type=Path);ap.add_argument('reference_weights',type=Path);ap.add_argument('reference_repo',type=Path);ap.add_argument('native_output',type=Path);ap.add_argument('--mimi',type=Path);ap.add_argument('--f16-weights',action='store_true');a=ap.parse_args()
sys.path.insert(0,str(a.reference_repo));torch.set_num_threads(6)
package=types.ModuleType("dia2");package.__path__=[str(a.reference_repo/"dia2")];sys.modules["dia2"]=package
from dia2.config import load_config
from dia2.core.model import Dia2Model
from dia2.core.precision import Precision
config=load_config(a.package/'config.json');precision=Precision(torch.float32,torch.float32)
from transformers import AutoTokenizer
from dia2.runtime.script_parser import parse_script
tokenizer=AutoTokenizer.from_pretrained(str(a.package),local_files_only=True)
constants=types.SimpleNamespace(spk1=49152,spk2=49153)
entries=parse_script(['[S1] Hello there! [S2] Does this work?'],tokenizer,constants,12.5)
expected=''.join(e.text+'\t'+str(e.padding)+''.join('\t'+str(i) for i in e.tokens)+'\n' for e in entries)
assert (a.native_output/'entries.tsv').read_text()==expected,'Tokenizer/script parsing parity failed'
print('PASS: tokenizer and speaker-marker parsing',flush=True)
model=Dia2Model(config,precision,device=torch.device('cpu'))
state=model.state_dict()
with safe_open(str(a.reference_weights),framework='pt',device='cpu') as f:
 for key in f.keys():
  v=f.get_tensor(key)
  if a.f16_weights and v.ndim>=2:v=v.half().float()
  state[key].copy_(v)
model.eval();cache=model.init_state(2,torch.device('cpu'),4);results=[]
def check(name,value):
 native=np.fromfile(a.native_output/name,dtype=np.float32)
 if name.startswith('dep-'):native=native.reshape(2,2050)[:,:2048].reshape(-1)
 ref=value.detach().float().numpy().reshape(-1)
 if native.shape!=ref.shape:raise AssertionError((name,native.shape,ref.shape))
 diff=np.abs(native-ref);cos=float(np.dot(native,ref)/(np.linalg.norm(native)*np.linalg.norm(ref)))
 record={'name':name,'max_abs':float(diff.max()),'rms':float(np.sqrt(np.mean(diff**2))),'cosine':cos}
 print(json.dumps(record),flush=True);results.append(record)
 if not np.all(np.isfinite(native)) or cos<0.999 or diff.max()>0.04:raise AssertionError(f'Parity failed: {record}')
with torch.inference_mode():
 for step in range(3):
  tokens=torch.empty((2,34,1),dtype=torch.long);tokens[:,0,0]=torch.tensor([10+step if step else 1,7]);tokens[:,1,0]=torch.tensor([11+step if step else 3,3]);tokens[:,2:,0]=torch.arange(100,132) if step else 2048
  h,action,cb0=model.step_text(tokens,torch.tensor([[step],[step]]),cache)
  check(f'hidden-{step}.f32',h);check(f'action-{step}.f32',action);check(f'cb0-{step}.f32',cb0)
 cache.depformer=model.depformer.init_cache(2,torch.device('cpu'),31)
 for stage in range(31):
  logits=model.step_audio_stage(stage,torch.tensor([123+stage]*2),h,cache,None,None)
  check(f'dep-{stage}.f32',logits)
if a.mimi:
 from transformers import MimiModel
 mimi=MimiModel.from_pretrained(str(a.mimi),local_files_only=True).eval()
 codes=torch.tensor([(i*37+41)%2048 for i in range(32*8)]).reshape(8,32).T.unsqueeze(0)
 with torch.inference_mode():audio=mimi.decode(codes,return_dict=False)[0]
 # F16 codec weights plus approximate erf/GELU can have a small waveform error.
 native=np.fromfile(a.native_output/'mimi.f32',dtype=np.float32);ref=audio.numpy().reshape(-1)
 cosine=float(np.dot(native,ref)/(np.linalg.norm(native)*np.linalg.norm(ref)));err=float(np.sqrt(np.mean((native-ref)**2)))
 record={'name':'mimi.f32','rms':err,'cosine':cosine};print(json.dumps(record),flush=True);results.append(record)
 if cosine<0.995 or err>0.01:raise AssertionError(f'Mimi parity failed: {record}')
 with torch.inference_mode():audio8=mimi.decode(codes[:,:8,:],return_dict=False)[0]
 native8=np.fromfile(a.native_output/'mimi8.f32',dtype=np.float32);ref8=audio8.numpy().reshape(-1)
 cosine8=float(np.dot(native8,ref8)/(np.linalg.norm(native8)*np.linalg.norm(ref8)));err8=float(np.sqrt(np.mean((native8-ref8)**2)))
 record={'name':'mimi8.f32','rms':err8,'cosine':cosine8};print(json.dumps(record),flush=True);results.append(record)
 if cosine8<0.995 or err8>0.01:raise AssertionError(f'Eight-codebook regression failed: {record}')
(a.native_output/'parity-report.json').write_text(json.dumps(results,indent=2)+'\n')
print('PASS: temporal transformer, all 31 depformer stages'+(', 32-codebook Mimi decoder' if a.mimi else ''),flush=True)
