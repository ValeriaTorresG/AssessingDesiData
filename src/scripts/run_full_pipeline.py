import argparse, subprocess, os, time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile', required=True)
    parser.add_argument('--night', required=True)
    parser.add_argument('--base-dir', default='./desi_data')
    parser.add_argument('--processed-dir', default='./data/processed')
    parser.add_argument('--band', default='brz', choices=['b','r','z','brz'])
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--n_neighbors', type=int, default=45)
    parser.add_argument('--min_dist', type=float, default=1.0)
    parser.add_argument('--n_components', type=int, default=2)
    parser.add_argument('--link-length', type=float, default=0.5)
    parser.add_argument('--min-cluster-size', type=int, default=5)
    args = parser.parse_args()

    time_start = time.time()
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.processed_dir+'/umap/', exist_ok=True)

    # 1) read data and save h5 per petal
    subprocess.run(['python', 'src/scripts/process_tile.py', '--tile', args.tile,
                    '--night', args.night, '--base-dir', args.base_dir,
                    '--out-dir', args.processed_dir], check=True)

    # 2) execute pad+UMAP+FoF per petal
    prefix = f'{args.processed_dir}/results_{args.night}/{args.tile}/umap/'
    cmd = ['python', 'src/scripts/run_model.py', args.processed_dir,
           '--night', args.night, '--tile', args.tile,
           '--band', args.band,
           '--out-prefix', prefix,
           '--n_neighbors', str(args.n_neighbors),
           '--min_dist',    str(args.min_dist),
           '--n_components',str(args.n_components),
           '--link-length', str(args.link_length),
           '--min-cluster-size', str(args.min_cluster_size)]
    if args.normalize:
        cmd.append('--normalize')
    subprocess.run(cmd, check=True)

    print('>>> Finished processing all petaaaaaals from tileeee', args.tile)
    print('>>> Total time taken:', time.time() - time_start, 'seconds')

if __name__ == '__main__':
    main()