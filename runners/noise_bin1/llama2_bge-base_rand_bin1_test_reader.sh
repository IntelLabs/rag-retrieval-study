
#!/bin/bash

SCRIPT_PATH="/home/aleto/projects/rag-svs/alce/"
export PYTHONPATH=${SCRIPT_PATH}

# ASQA
python3 reader/run.py \
    --config asqa_llama-2-7b-chat_shot1_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/asqa_bin0.json \
    --nrand 5 \
    --add_name bin0 \
    --noise_first

python3 reader/run.py \
    --config asqa_llama-2-7b-chat_shot1_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/asqa_bin1.json \
    --nrand 5 \
    --add_name bin1 \
    --noise_first

python3 reader/run.py \
    --config asqa_llama-2-7b-chat_shot1_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/asqa_bin0.json \
    --nrand 5 \
    --add_name bin0

python3 reader/run.py \
    --config asqa_llama-2-7b-chat_shot1_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/asqa_bin1.json\
    --nrand 5 \
    --add_name bin1

# QAMPARI
python3 reader/run.py \
    --config qampari_llama-2-7b-chat-shot1_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/qampari_bin0.json\
    --nrand 5 \
    --add_name bin0 \
    --noise_first

python3 reader/run.py \
    --config qampari_llama-2-7b-chat-shot1_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/qampari_bin1.json\
    --nrand 5 \
    --add_name bin1 \
    --noise_first

python3 reader/run.py \
    --config qampari_llama-2-7b-chat-shot1_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/qampari_bin0.json\
    --nrand 5 \
    --add_name bin0

python3 reader/run.py \
    --config qampari_llama-2-7b-chat-shot1_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/qampari_bin1.json\
    --nrand 5 \
    --add_name bin1


# NQ
python3 reader/run.py \
    --config nq_llama-2-7b-chat_shot1_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/nq_bin0.json \
    --nrand 5 \
    --add_name bin0

python3 reader/run.py \
    --config nq_llama-2-7b-chat_shot1_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/nq_bin0.json \
    --nrand 5 \
    --noise_first \
    --add_name bin0

python3 reader/run.py \
    --config nq_llama-2-7b-chat_shot1_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/nq_bin1.json \
    --nrand 5 \
    --add_name bin1

python3 reader/run.py \
    --config nq_llama-2-7b-chat_shot1_bge_base_default.yaml \
    --ndoc 5 \
    --noise_file noise_bins/nq_bin1.json \
    --nrand 5 \
    --noise_first \
    --add_name bin1
