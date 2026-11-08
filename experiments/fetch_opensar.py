"""Extract only Sentinel-1 VV SLC scene archives from OpenSARShip via HTTP ranges."""
import argparse
import io
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

BASE = "https://opensar.sjtu.edu.cn/openSAR/"


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        with urlopen(Request(url, method="HEAD"), timeout=30) as response:
            self.size = int(response.headers["Content-Length"])

    def seek(self, offset, whence=0):
        self.pos = (0, self.pos, self.size)[whence] + offset
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def read(self, count=-1):
        end = self.size if count < 0 else min(self.pos + count, self.size)
        if end <= self.pos:
            return b""
        request = Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=120) as response:
                    expected = f"bytes {self.pos}-{end - 1}/{self.size}"
                    if response.status != 206 or response.headers["Content-Range"] != expected:
                        raise ValueError("Server returned a different archive byte range")
                    result = response.read()
                if len(result) != end - self.pos:
                    raise IOError("Incomplete archive byte range")
            except (OSError, URLError):
                if attempt == 2:
                    raise
            else:
                self.pos = end
                return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--release", choices=("1", "2"), default="1")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-mb", type=float, default=float("inf"))
    parser.add_argument("--manifest", type=Path, help="Only download archives listed for this release")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    url = f"{BASE}OpenSARShip_{args.release}.zip"
    with ZipFile(RemoteZip(url)) as outer:
        members = sorted(
            (m for m in outer.infolist() if "SLC__1SDV" in m.filename and m.filename.endswith(".zip")),
            key=lambda m: (m.file_size, m.filename),
        )
    members = [m for m in members if m.file_size <= args.max_mb * 1e6]
    if args.manifest:
        names = {Path(line).name for line in args.manifest.read_text().splitlines()
                 if Path(line).parent.name == f"opensar{args.release}"}
        members = [m for m in members if Path(m.filename).name in names]
        if {Path(m.filename).name for m in members} != names:
            raise ValueError("Some manifest members were missing from the release")
    if args.list:
        for m in members:
            print(m.filename, m.file_size)
        return
    args.destination.mkdir(parents=True, exist_ok=True)

    def extract(member):
        dest = args.destination / Path(member.filename).name
        if dest.exists():
            return
        tmp = dest.with_suffix(".part")
        with ZipFile(RemoteZip(url)) as outer, outer.open(member) as source, tmp.open("wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
        tmp.replace(dest)
        print(dest.name, flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(extract, members))


if __name__ == "__main__":
    main()
