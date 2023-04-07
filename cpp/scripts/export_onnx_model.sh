#!/bin/bash

export PYTHONPATH=/root/wenet-onnx/wenet  # the directory of wenet root.
export CUDA_VISIBLE_DEVICES="1"  # gpu id
FP16=--fp16  
SRCDIR=/root/wenet-onnx/models/$1 #the directory of source path of checkpoint models.
OUTDIR=onnx_$1
python export_onnx.py  --config $SRCDIR/train.yaml  --checkpoint $SRCDIR/final.pt  --cmvn_file $SRCDIR/global_cmvn  $FP16 --output_onnx_dir $OUTDIR