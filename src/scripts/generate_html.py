import os
import re
from typing import Optional
from jinja2 import Environment, FileSystemLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
FIBER_DIR = os.path.join(DATA_ROOT, 'plots', 'fibers')
SPECTRA_DIR = os.path.join(DATA_ROOT, 'plots', 'spectra')
UMAP_DIR = os.path.join(DATA_ROOT, 'plots', 'umap')
TXT_DIR = os.path.join(DATA_ROOT, 'text_files')

TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'web_view')
OUTPUT_HTML = os.path.join(PROJECT_ROOT, 'index.html')

BASE_URL = 'data'

def make_desi_url(tile: str, night: str) -> Optional[str]:
    txt_name = f'{night}_{tile}.txt'
    txt_path = os.path.join(TXT_DIR, txt_name)
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
    return f'https://inspector.desi.lbl.gov/loa/spectra/tiles/{tile}/{fiber_str}'


def generate_html():
    pattern = re.compile(r'fibers_(\d{8})_(\d+)\.png')
    entries = []

    for fname in sorted(os.listdir(FIBER_DIR)):
        m = pattern.match(fname)
        if m:
            night, tile = m.group(1), m.group(2)
            spec_dir = os.path.join(SPECTRA_DIR, night)
            specs = []
            if os.path.isdir(spec_dir):
                specs = sorted(f for f in os.listdir(spec_dir)
                    if f.startswith(f'spec_{tile}_') and f.lower().endswith('.png')
                    )

            inspector_url = make_desi_url(tile, night)

            entries.append({'tile':tile, 'night':night, 'spectra':specs,
                            'inspector_url':inspector_url
                            })

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)
    tpl = env.get_template('template.html')
    html = tpl.render(entries=entries, base_url=BASE_URL)

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)

if __name__ == '__main__':
    generate_html()