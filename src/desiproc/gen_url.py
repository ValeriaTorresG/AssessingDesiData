import os

def make_desi_url(txt_dir:str, tile:str, night:str, out_log:str):
    txt_name = f'{night}_{tile}.txt'
    txt_path = os.path.join(txt_dir, txt_name)
    if not os.path.isfile(txt_path):
        return None

    fibers = set()
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('TARGETID'):
                continue
            parts = [p.strip() for p in line.split(',')]
            _, t, fiber = parts[0], parts[1], parts[2]
            fibers.add(int(fiber))

    fiber_str = ','.join(str(f) for f in sorted(fibers))
    url = f'https://inspector.desi.lbl.gov/loa/spectra/tiles/{tile}/{fiber_str}'
    
    with open(out_log, 'a', encoding='utf-8') as f:
        f.write(f'{night},{tile},{url}\n')