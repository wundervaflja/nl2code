output="runs"
device="cpu"

if [ "$1" == "hs" ]; then
    # hs dataset
    echo "run trained model for hs"
    dataset="data/hs.freq3.pre_suf.unary_closure.bin"
    model="model.hs_unary_closure_top20_word128_encoder256_rule128_node64.beam15.adadelta.simple_trans.8e39832.iter5600.npz"
    commandline="-decode_max_time_step 350 -rule_embed_dim 128 -node_embed_dim 64"
    datatype="hs"
else
    # django dataset
    echo "run trained model for django"
    dataset="data/django.cleaned.dataset.freq3.par_info.refact.space_only.order_by_ulink_len.bin"
    model="model.best_acc.npz"
    commandline="-rule_embed_dim 128 -node_embed_dim 64"
    datatype="django"
fi

# run interactive mode on trained models
THEANO_FLAGS="mode=FAST_RUN,device=${device},floatX=float32" python nl2code/code_gen.py \
    -data_type ${datatype} \
    -data nl2code/${dataset} \
    -output_dir nl2code/${output} \
    -model nl2code/runs/${model} \
    ${commandline} \
    interactive \
    -mode new
