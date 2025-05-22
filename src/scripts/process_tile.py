from concurrent.futures import ProcessPoolExecutor, as_completed  #maybe better than ThreadPoolExecutor in this case
from pathlib import Path
import argparse, time
from desiproc.save_data import SpectraPipeline

def process_petal(petal: int, args) -> None:
    coadd = Path(args.base_dir)/args.tile/args.night/f"coadd-{petal}-{args.tile}-thru{args.night}.fits"
    rr = Path(args.base_dir)/args.tile/args.night/f"redrock-{petal}-{args.tile}-thru{args.night}.fits"
    SpectraPipeline(str(coadd), str(rr),  data_path=args.out_dir).write_data()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--night',    required=True)
    parser.add_argument('--tile',     required=True)
    parser.add_argument('--base-dir', required=True)
    parser.add_argument('--out-dir',  required=True)
    parser.add_argument('--workers',  type=int, default=10)
    args = parser.parse_args(argv)

    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = [exe.submit(process_petal, petal, args) for petal in range(10)]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                continue