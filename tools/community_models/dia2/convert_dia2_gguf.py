#!/usr/bin/env python3
"""Convert official Dia2 weights and HF Mimi into audio.cpp GGUF packages."""
import argparse, json, re, shutil
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from gguf import GGUFWriter, GGMLQuantizationType, GGUFValueType, LlamaFileType, quantize

def write(source, output, architecture, precision, config=None, mimi=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(output), architecture, use_temp_file=True)
    writer.add_name('Dia2' if not mimi else 'Kyutai Mimi')
    writer.add_string('general.license', 'apache-2.0' if not mimi else 'cc-by-4.0')
    file_type = (LlamaFileType.MOSTLY_Q8_0 if precision.startswith('q8') else
                 LlamaFileType.MOSTLY_Q4_0 if precision.startswith('q4') else
                 LlamaFileType.ALL_F32 if precision == 'f32' else LlamaFileType.MOSTLY_F16)
    writer.add_file_type(file_type)
    if precision.startswith(('q8', 'q4')):
        writer.add_quantization_version(2)
    if config: writer.add_string('dia2.config', json.dumps(config))
    count = 0; names=[]; ranks=[]; shapes=[]
    with safe_open(str(source), framework='pt', device='cpu') as f:
        for key in f.keys():
            if mimi and any(key.endswith('self_attn.'+p+'.weight') for p in ('k_proj','v_proj')): continue
            arr = f.get_tensor(key).float().numpy()
            name = key
            if mimi:
                if '.self_attn.q_proj.weight' in key:
                    prefix=key.split('.self_attn.')[0]
                    # HF uses half-split RoPE; shared Mimi uses interleaved RoPE.
                    def interleave(a):
                        return a.reshape(8,2,32,512).transpose(0,2,1,3).reshape(512,512)
                    arr=np.concatenate([interleave(arr),interleave(f.get_tensor(prefix+'.self_attn.k_proj.weight').numpy()),f.get_tensor(prefix+'.self_attn.v_proj.weight').numpy()])
                    name=prefix+'.self_attn.in_proj_weight'
                name=name.replace('encoder.layers.', 'encoder.model.').replace('decoder.layers.', 'decoder.model.')
                name=name.replace('encoder_transformer.layers.','encoder_tf.layers.').replace('decoder_transformer.layers.','decoder_tf.layers.')
                name=name.replace('input_layernorm','norm1').replace('post_attention_layernorm','norm2').replace('mlp.fc1','linear1').replace('mlp.fc2','linear2').replace('self_attn.o_proj','self_attn.out_proj').replace('self_attn_layer_scale','layer_scale_1').replace('mlp_layer_scale','layer_scale_2')
                name=name.replace('quantizer.semantic_residual_vector_quantizer','quantizer.rvq_first').replace('quantizer.acoustic_residual_vector_quantizer','quantizer.rvq_rest').replace('.layers.', '.vq.layers.') if name.startswith('quantizer.') else name
                name=name.replace('.codebook.', '._codebook.').replace('.embed_sum', '.embedding_sum')
                if name.startswith(('encoder.model.','decoder.model.')):
                    is_transpose=name.startswith('decoder.model.') and int(name.split('.')[2]) in (2,5,8,11) and '.block.' not in name
                    name=name.replace('.conv.', '.convtr.convtr.' if is_transpose else '.conv.conv.')
                name=name.replace('downsample.conv.weight','downsample.conv.conv.conv.weight').replace('upsample.conv.weight','upsample.convtr.convtr.convtr.weight')
            names.append(name); ranks.append(arr.ndim); shapes.extend(arr.shape)
            arr=np.ascontiguousarray(arr)
            qtype=GGMLQuantizationType.F32
            if arr.ndim==2 and not mimi and precision in ('q8', 'q8_0', 'q4', 'q4_0'):
                qtype = GGMLQuantizationType.Q8_0 if precision.startswith('q8') else GGMLQuantizationType.Q4_0
                if arr.shape[-1] % 32:
                    raise ValueError(f'{key}: row width must be divisible by 32 for {qtype.name}')
                arr=quantize(arr, qtype)
            elif arr.ndim>=2 and precision!='f32':
                arr=arr.astype(np.float16); qtype=GGMLQuantizationType.F16
            writer.add_tensor(f"t{count:04d}",arr,raw_dtype=qtype)
            count+=1
    writer.add_array("audiocpp.tensor_names",names)
    writer.add_key_value("audiocpp.tensor_ranks",ranks,GGUFValueType.ARRAY,sub_type=GGUFValueType.INT32)
    writer.add_key_value("audiocpp.tensor_shapes",shapes,GGUFValueType.ARRAY,sub_type=GGUFValueType.INT64)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    print(f'{output}: {count} tensors, {output.stat().st_size/1e9:.3f} GB',flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('model',type=Path); ap.add_argument('output',type=Path); ap.add_argument('--precision',choices=['f32','f16','q8','q8_0','q4','q4_0'],default='f16'); ap.add_argument('--mimi',type=Path)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    config=json.loads((a.model/'config.json').read_text())
    write(a.model/'model.safetensors',a.output/f'dia2-{a.precision}.gguf','dia2',a.precision,config)
    for name in ['config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json','added_tokens.json']:
        if (a.model/name).exists(): shutil.copy2(a.model/name,a.output/name)
    if a.mimi: write(a.mimi/'model.safetensors',a.output/'mimi-f16.gguf','mimi','f16',mimi=True)
if __name__=='__main__': main()
