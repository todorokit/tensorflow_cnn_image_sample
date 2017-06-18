export CUDA_VISIBLE_DEVICES=$1
shift

echo "gpu set to $CUDA_VISIBLE_DEVICES;"
$@
