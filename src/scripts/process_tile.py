import argparse, time, sys, os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from desiproc.save_data import SpectraPipeline

def process_petal(petal, tile, night, base_dir, out_dir):
    fn_coadd = f'{base_dir}/{tile}/{night}/coadd-{petal}-{tile}-thru{night}.fits'
    fn_rr    = f'{base_dir}/{tile}/{night}/redrock-{petal}-{tile}-thru{night}.fits'
    pipeline = SpectraPipeline(fn_coadd, fn_rr, data_path=out_dir)
    pipeline.write_data()
    return petal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--night',required=True, help='Night')
    parser.add_argument('--tile', required=True, help='Tile ID')
    parser.add_argument('--base-dir',default='./desi_data', help='Path of coadd/redrock')
    parser.add_argument('--out-dir', default='./data/processed', help='Path for .h5 outputs')
    parser.add_argument('--workers', type=int, default=10, help='Max concurrent threads')
    args = parser.parse_args()

    start_all = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(process_petal, p, args.tile, args.night, args.base_dir,
                              args.out_dir): p for p in range(10)}
        for fut in as_completed(futures):
            petal = futures[fut]
    # print(f'>>> Tile processed in {time.time()-start_all:.2f} s')

if __name__ == '__main__':
    main()