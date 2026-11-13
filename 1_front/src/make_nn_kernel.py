"""신경망 커널: CONFIG_OVERRIDE + sincon_v2.py(메인 비활성) + nn_mlp.py. 사용: python make_nn_kernel.py <slug> '<json>'"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
slug, cfg = sys.argv[1], json.loads(sys.argv[2])
d = ROOT / "kernels" / slug
d.mkdir(parents=True, exist_ok=True)
src = (ROOT / "src" / "sincon_v2.py").read_text(encoding="utf-8").replace('if __name__ == "__main__":', "if False:")
nn = (ROOT / "src" / "nn_mlp.py").read_text(encoding="utf-8")
(d / "run.py").write_text(f"CONFIG_OVERRIDE = {cfg!r}\n" + src + "\n" + nn, encoding="utf-8")
meta = {"id": f"rkdaltmd/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": False, "dataset_sources": ["rkdaltmd/sincon-work"],
        "competition_sources": [], "kernel_sources": []}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print(d)
