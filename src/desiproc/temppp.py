import os
from gen_url_fiber import make_url_fiber

folder = '/pscratch/sd/v/vtorresg/umap_analysis/data/text_files'
files = os.listdir(folder)
for file in files:
    n = file.split('_')[0]
    make_url_fiber(folder, n,
                   f'/pscratch/sd/v/vtorresg/umap_analysis/data/inspector_urls_fib.txt')