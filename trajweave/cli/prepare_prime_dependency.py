"""为项目环境准备 PRIME 可选依赖，兼容 Python 3.11。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sysconfig
import tarfile
from urllib.request import urlopen


URL = "https://files.pythonhosted.org/packages/9c/8e/75e816409b51bce77ec0b99073b462f028509c65a26bfdc87ed88f712dc9/pyext-0.6.tar.gz"
SHA256 = "6c406cf71b991e1fc5a7f963d3a289525bce5e7ad1c43b697d9f5223185fcaef"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path(sysconfig.get_path("purelib")))
    args = parser.parse_args(argv)
    with urlopen(URL, timeout=60) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != SHA256:
        raise RuntimeError("pyext 源码校验失败，未写入文件")
    # 仅读取指定成员，不解压归档中的路径或执行 setup.py。
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
        source = package.extractfile("pyext-0.6/pyext.py").read().decode()
    source = source.replace("oargspec = inspect.getargspec",
                            'oargspec = getattr(inspect, "getargspec", inspect.getfullargspec)')
    compile(source, "pyext.py", "exec")
    args.target.mkdir(parents=True, exist_ok=True)
    destination = args.target / "pyext.py"
    if destination.exists() and destination.read_text() != source:
        raise FileExistsError(f"保留现有不同版本，请指定独立 --target：{destination}")
    destination.write_text(source)
    metadata = {"source": URL, "archive_sha256": SHA256,
                "installed_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "compatibility_patch": "inspect.getargspec fallback to getfullargspec"}
    (args.target / "trajweave-pyext-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"module": str(destination), **metadata}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
