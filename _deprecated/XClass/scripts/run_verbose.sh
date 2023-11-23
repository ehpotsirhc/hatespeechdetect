set -e

gpu=$1
dataset=$2

printf ">>> python static_representations.py\n"
CUDA_VISIBLE_DEVICES=${gpu} python static_representations.py --dataset_name ${dataset}

printf ">>> class_oriented_document_representations.py\n"
CUDA_VISIBLE_DEVICES=${gpu} python class_oriented_document_representations.py --dataset_name ${dataset}

printf ">>> document_class_alignment.py\n"
python document_class_alignment.py --dataset_name ${dataset}

printf ">>> evaluate.py\n"
python evaluate.py --dataset ${dataset} --stage Rep --suffix bbu-12-mixture-100

printf ">>> evaluate.py\n"
python evaluate.py --dataset ${dataset} --stage Align --suffix pca64.clusgmm.bbu-12.mixture-100.42

printf ">>> prepare_text_classifer_training.py\n"
python prepare_text_classifer_training.py --dataset_name ${dataset}

printf ">>> ./run_train_text_classifier.sh\n"
./run_train_text_classifier.sh ${gpu} ${dataset} pca64.clusgmm.bbu-12.mixture-100.42.0.5

printf ">>> evaluate.py [Rep --suffix bbu-12-mixture-100]\n"
python evaluate.py --dataset ${dataset} --stage Rep --suffix bbu-12-mixture-100

printf ">>> evaluate.py [Align --suffix pca64.clusgmm.bbu-12.mixture-100.42\n"
python evaluate.py --dataset ${dataset} --stage Align --suffix pca64.clusgmm.bbu-12.mixture-100.42
