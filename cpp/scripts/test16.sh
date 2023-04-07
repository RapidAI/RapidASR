#!/bin/bash
  
FP16=--fp16
MODELDIR=/root/wenet-onnx/wenet/wenet/bin/models/onnx_20211025_conformer_exp
python recognize_onnx.py  --test_data $1 --dict $MODELDIR/words.txt --config $MODELDIR/train.yaml --encoder_onnx $MODELDIR/encoder_fp16.onnx --decoder_onnx $MODELDIR/decoder_fp16.onnx --result_file result.txt  $FP16
