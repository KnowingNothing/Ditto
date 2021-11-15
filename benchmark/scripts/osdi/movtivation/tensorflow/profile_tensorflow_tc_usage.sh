export CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.1
ncu --metrics "sm__cycles_active.sum,smsp__pipe_tensor_cycles_active.sum" --print-summary per-gpu --csv python tensorflow_network.py --dnn mobile --batch 1 --dtype mix --use_xla |& tee tensorflow_mobilenet_b1_mix_profile.csv
ncu --metrics "sm__cycles_active.sum,smsp__pipe_tensor_cycles_active.sum" --print-summary per-gpu --csv python tensorflow_network.py --dnn mobile --batch 16 --dtype mix --use_xla |& tee tensorflow_mobilenet_b16_mix_profile.csv
ncu --metrics "sm__cycles_active.sum,smsp__pipe_tensor_cycles_active.sum" --print-summary per-gpu --csv python tensorflow_network.py --dnn res50 --batch 1 --dtype mix --use_xla |& tee tensorflow_resnet50_b1_mix_profile.csv
ncu --metrics "sm__cycles_active.sum,smsp__pipe_tensor_cycles_active.sum" --print-summary per-gpu --csv python tensorflow_network.py --dnn res50 --batch 16 --dtype mix --use_xla |& tee tensorflow_resnet50_b16_mix_profile.csv
# can't get results for nasnet
# ncu --metrics "sm__cycles_active.sum,smsp__pipe_tensor_cycles_active.sum" --print-summary per-gpu --csv python tensorflow_network.py --dnn nas --batch 1 --dtype mix --use_xla |& tee tensorflow_nasnet_b1_mix_profile.csv
# ncu --metrics "sm__cycles_active.sum,smsp__pipe_tensor_cycles_active.sum" --print-summary per-gpu --csv python tensorflow_network.py --dnn nas --batch 16 --dtype mix --use_xla |& tee tensorflow_nasnet_b16_mix_profile.csv
ncu --metrics "sm__cycles_active.sum,smsp__pipe_tensor_cycles_active.sum" --print-summary per-gpu --csv python tensorflow_network.py --dnn xception --batch 1 --dtype mix --use_xla |& tee tensorflow_xception_b1_mix_profile.csv
ncu --metrics "sm__cycles_active.sum,smsp__pipe_tensor_cycles_active.sum" --print-summary per-gpu --csv python tensorflow_network.py --dnn xception --batch 16 --dtype mix --use_xla |& tee tensorflow_xception_b16_mix_profile.csv
ncu --metrics "sm__cycles_active.sum,smsp__pipe_tensor_cycles_active.sum" --print-summary per-gpu --csv python tensorflow_network.py --dnn bert --batch 1 --dtype mix --use_xla |& tee tensorflow_bert_b1_mix_profile.csv
ncu --metrics "sm__cycles_active.sum,smsp__pipe_tensor_cycles_active.sum" --print-summary per-gpu --csv python tensorflow_network.py --dnn mi-lstm --batch 1 --dtype mix --use_xla |& tee tensorflow_milstm_b1_mix_profile.csv
ncu --metrics "sm__cycles_active.sum,smsp__pipe_tensor_cycles_active.sum" --print-summary per-gpu --csv python tensorflow_network.py --dnn mi-lstm --batch 16 --dtype mix --use_xla |& tee tensorflow_milstm_b16_mix_profile.csv
ncu --metrics "sm__cycles_active.sum,smsp__pipe_tensor_cycles_active.sum" --print-summary per-gpu --csv python tensorflow_network.py --dnn shuffle --batch 1 --dtype mix --use_xla |& tee tensorflow_shufflenet_b1_mix_profile.csv
ncu --metrics "sm__cycles_active.sum,smsp__pipe_tensor_cycles_active.sum" --print-summary per-gpu --csv python tensorflow_network.py --dnn shuffle --batch 16 --dtype mix --use_xla |& tee tensorflow_shufflenet_b16_mix_profile.csv
