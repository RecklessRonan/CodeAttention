WORKDIR="/data/pretrain-attention/CodeAttention"
export PYTHONPATH=$WORKDIR


MODEL_NAME=${1}
TASK=${2}
SUB_TASK=${3}

DATA_NUM=-1
MODEL_DIR=save_models
SUMMARY_DIR=tensorboard
FULL_MODEL_TAG=${MODEL_NAME}

if [[ ${SUB_TASK} == none ]]; then
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${FULL_MODEL_TAG}
  RES_DIR=results/${TASK}/${FULL_MODEL_TAG}
  RES_FN=results/${TASK}/${FULL_MODEL_TAG}.txt
  ATTENTION_DIR=attentions/${TASK}/${FULL_MODEL_TAG}
else
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
  RES_DIR=results/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
  RES_FN=results/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}.txt
  ATTENTION_DIR=attentions/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
fi

CACHE_DIR=${OUTPUT_DIR}/cache_data
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}
mkdir -p ${ATTENTION_DIR}
LOG=${ATTENTION_DIR}/log.txt


RUN_FN=${WORKDIR}/attention.py

CUDA_VISIBLE_DEVICES=0 \
TOKENIZERS_PARALLELISM=false \
  python ${RUN_FN}\
  --do_test --do_train --do_eval --do_eval_bleu --save_last_checkpoints --always_save_model \
  --task ${TASK} --sub_task ${SUB_TASK} --model_name ${MODEL_NAME} --data_num ${DATA_NUM}  \
  --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
  --data_dir ${WORKDIR}/data  --cache_path ${CACHE_DIR} --res_dir ${RES_DIR} --res_fn ${RES_FN} \
  2>&1 | tee ${LOG}
