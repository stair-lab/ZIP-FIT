# - Args
export mode='dryrun'
export mode='online'

# - Run
export CUDA_VISIBLE_DEVICES=0
conda activate zip_fit
python -m zip_fit.metrics.tests.tests_log_likelihoods_gold_ref --mode=$mode
