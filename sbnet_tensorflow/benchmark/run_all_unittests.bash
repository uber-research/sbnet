export TF_CPP_MIN_LOG_LEVEL=2
python -m unittest discover -v -s . -p "*_tests.py"
#python -m unittest reduce_mask_tests.ReduceMaskTests
#python -m unittest sparse_scatter_tests.SparseScatterTests
