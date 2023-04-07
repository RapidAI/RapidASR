#!/bin/bash
  

MODELDIR=/root/wenet-onnx/wenet/wenet/bin/models/onnx_20211025_conformer_exp
python recognize_onnx.py  --test_data $1 --dict $MODELDIR/words.txt --config $MODELDIR/train.yaml --encoder_onnx $MODELDIR/encoder.onnx --decoder_onnx $MODELDIR/decoder.onnx --result_file result.txt  
~                                                                                                                                                                                                                
