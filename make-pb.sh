# git clone 
#TENSORFLOW_DIR=~/src/tensorflow/
TENSORFLOW_DIR=~/.pyenv/versions/3.6.6/lib/python3.6/site-packages/tensorflow/python/tools/freeze_graph.py
if [ -n "$1" ] ; then
    python make-pb.py --config $1
else
    python make-pb.py
fi

python $TENSORFLOW_DIR/tensorflow/python/tools/freeze_graph.py \
       --input_graph=/tmp/graph.pb --input_checkpoint=model_v3.ckpt \
       --output_graph=/tmp/frozen_graph.pb \
       --output_node_names=fp32_storage/tower_0/fc/fc_var/fc_act \
       --clear_devices=True --input_binary=True
