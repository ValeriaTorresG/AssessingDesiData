from pathlib import Path
from collections import defaultdict

def make_url_fiber(txt_dir: str, night: str, out_log: str):
    txt_path = Path(txt_dir)
    prefix = f"{night}_"
    fiber_to_targets = defaultdict(set)

    for file in txt_path.iterdir():
        name = file.name
        if not (name.startswith(prefix) and name.endswith(".txt")):
            continue

        with file.open("r", encoding="utf-8") as f:
            add = fiber_to_targets  # alias local
            for line in f:
                if not line or line[0] == "T":
                    continue
                targetid, _, rest   = line.partition(',')
                _, _, fiber_str     = rest.rpartition(',')
                add[int(fiber_str)].add(targetid)

    lines = []
    base = "https://inspector.desi.lbl.gov/loa/spectra"
    for fiber in sorted(fiber_to_targets):
        targets = sorted(fiber_to_targets[fiber], key=int)
        url     = f"{base}/{','.join(targets)}"
        lines.append(f"{fiber}: {url}\n")

    Path(out_log).write_text(''.join(lines), encoding="utf-8")