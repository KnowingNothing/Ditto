# conv_conv_oneDNN.pkl
python ./oneDNN/conv_conv.py --dtype float32 --server Xeon-Gold-6348
# conv_conv_torch.pkl
python ./pytorch/cpu/conv_conv.py --server Xeon-Gold-6348
# conv_conv_nchwc_chimera.pkl
python ./chimera/cpu/conv_conv_nchwc.py --server Xeon-Gold-6348 --mode best
# conv_conv_relay.pkl
python ./tvm/cpu/conv_conv_relay.py
# conv_conv_ansor.pkl
python ./tvm/cpu/conv_conv_ansor.py --server Xeon-Gold-6348

# bmm_bmm_onednn.pkl
python ./oneDNN/bmm_bmm.py --dtype float32 --server Xeon-Gold-6348
# bmm_bmm_torch.pkl error
python ./pytorch/cpu/bmm_bmm.py --server Xeon-Gold-6348
# bmm_bmm_chimera.pkl 
python ./chimera/cpu/bmm_bmm_cpu.py --server Xeon-Gold-6348 --mode best
# bmm_bmm_relay.pkl error
python ./tvm/cpu/bmm_bmm_relay.py --server Xeon-Gold-6348
# bmm_bmm_ansor.pkl
python ./tvm/cpus/bmm_bmm_ansor.py --server Xeon-Gold-6348

# conv_relu_conv_oneDNN.pkl
python ./oneDNN/conv_relu_conv.py --dtype float32 --server Xeon-Gold-6348
# conv_relu_conv_torch.pkl bad
python ./pytorch/cpu/conv_relu_conv.py --server Xeon-Gold-6348
# conv_relu_conv_nchwc_chimera.pkl
python ./chimera/cpu/conv_relu_conv_nchwc.py --server Xeon-Gold-6348 --mode best
# conv_relu_conv_ansor.pkl
python ./tvm/cpu/conv_relu_conv_ansor.py
# conv_relu_conv_relay.pkl
python ./tvm/cpu/conv_relu_conv_relay.py

# bmm_softmax_bmm_onednn.pkl
python ./oneDNN/bmm_softmax_bmm.py --dtype float32 --server Xeon-Gold-6348
# bmm_softmax_bmm_torch.pkl 
python ./pytorch/cpu/bmm_softmax_bmm.py --server Xeon-Gold-6348
# bmm_softmax_bmm_chimera.pkl 
python ./chimera/cpu/bmm_softmax_bmm.py --server Xeon-Gold-6348 --mode best
# bmm_softmax_bmm_relay.pkl error
python ./tvm/cpu/bmm_softmax_bmm_relay.py --server Xeon-Gold-6348
# bmm_softmax_bmm_ansor.pkl error
python ./tvm/cpu/bmm_softmax_bmm_ansor.py --server Xeon-Gold-6348
