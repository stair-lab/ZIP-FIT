# - Args
export mode='dryrun'
export mode='online'

# - Run
python -m zip_fit.metrics.tests.tests_tfa --mode=$mode

# indeed the above reproduced previous results:
# prevvious result: https://stackoverflow.com/questions/79209319/how-to-compute-teacher-forced-accuracy-tfa-for-hugging-face-models-while-handl/79379540#79379540
# new result: https://wandb.ai/brando/zip-fit-tfa-tests/runs/sxi8y9w3/overview 