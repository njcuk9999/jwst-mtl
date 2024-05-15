from loicpipe.core import general
from loicpipe.loicpipe import stage1

# load parameters
params = general.load_params()

# verify data
# general.verify_data(params)

# run stage 1
params = stage1.main(params)
