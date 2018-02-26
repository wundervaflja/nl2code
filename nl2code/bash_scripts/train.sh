output="runs"
device="cpu"

if [ "$1" == "hs" ]; then
	# hs dataset
	echo "training hs dataset"
	dataset="hs.freq3.pre_suf.unary_closure.bin"
	commandline="-batch_size 10 -max_epoch 200 -valid_per_batch 280 -save_per_batch 280 -decode_max_time_step 350 -optimizer adadelta -rule_embed_dim 128 -node_embed_dim 64 -valid_metric bleu"
	datatype="hs"
else
	# django dataset
	echo "training django dataset"
	dataset="django.cleaned.dataset.freq3.par_info.refact.space_only.order_by_ulink_len.bin"
	commandline="-batch_size 10 -max_epoch 1 -valid_per_batch 4000 -save_per_batch 4000 -decode_max_time_step 100 -optimizer adam -rule_embed_dim 128 -node_embed_dim 64 -valid_metric bleu"
	datatype="django"
fi

# train the model
THEANO_FLAGS="mode=FAST_RUN,device=${device},floatX=float32" python -u nl2code/code_gen.py \
	-data_type ${datatype} \
	-data nl2code/data/${dataset} \
	-output_dir nl2code/${output} \
	${commandline} \
	train

# # decode testing set, and evaluate the model which achieves the best bleu and accuracy, resp.
# for model in "model.best_bleu.npz" "model.best_acc.npz"; do
# 	THEANO_FLAGS="mode=FAST_RUN,device=${device},floatX=float32" python nl2code/code_gen.py \
# 	-data_type ${datatype} \
# 	-data nl2code/data/${dataset} \
# 	-output_dir nl2code/${output} \
# 	-model nl2code/${output}/${model} \
# 	${commandline} \
# 	decode \
# 	-saveto ${output}/${model}.decode_results.test.bin

# 	python nl2code/code_gen.py \
# 		-data_type ${datatype} \
# 		-data nl2code/data/${dataset} \
# 		-output_dir nl2code/${output} \
# 		evaluate \
# 		-input nl2code/${output}/${model}.decode_results.test.bin
# done
