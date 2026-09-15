"""Read the AddBiomechanics 417 GB zip over HTTP without downloading it.

Zip is a random-access format: the central directory sits at the end of the file,
so with HTTP range requests we can list every member and pull out only the few
we actually need (the Hamner2013 running subjects).
"""
import io
import os
import sys
import time
import zipfile

import requests

URL = "http://archive.simtk.org/addbiomechanics/addbiomechanics.zip"
THROTTLE = float(os.environ.get("ZIP_THROTTLE", "0.3"))  # seconds between range gets


class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable file-like object backed by HTTP range requests."""

    def __init__(self, url, session=None, block=1 << 20):
        self.url = url
        self.sess = session or requests.Session()
        self.block = block
        self.pos = 0
        head = self.sess.head(url, allow_redirects=True, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        if head.headers.get("Accept-Ranges") != "bytes":
            raise RuntimeError("server does not advertise byte ranges")
        self._cache = {}

    # --- io plumbing -----------------------------------------------------
    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self.pos = offset
        elif whence == os.SEEK_CUR:
            self.pos += offset
        elif whence == os.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    # --- the actual fetching ---------------------------------------------
    def _fetch(self, start, end):
        """Fetch [start, end] inclusive, caching on `block`-aligned chunks."""
        out = bytearray()
        first, last = start // self.block, end // self.block
        for idx in range(first, last + 1):
            if idx not in self._cache:
                lo = idx * self.block
                hi = min(lo + self.block, self.size) - 1
                # simtk throttles bursts of range requests with 403s; back off
                for attempt in range(12):
                    try:
                        r = self.sess.get(
                            self.url,
                            headers={"Range": f"bytes={lo}-{hi}"},
                            timeout=120,
                        )
                        if r.status_code in (403, 429, 500, 502, 503, 504):
                            raise requests.HTTPError(f"status {r.status_code}")
                        r.raise_for_status()
                        break
                    except (requests.RequestException, requests.HTTPError) as exc:
                        wait = min(60, 2 ** attempt)
                        print(
                            f"    retry {attempt + 1}/12 after {exc} (sleep {wait}s)",
                            flush=True,
                        )
                        time.sleep(wait)
                else:
                    raise RuntimeError(f"gave up fetching bytes {lo}-{hi}")
                self._cache[idx] = r.content
                time.sleep(THROTTLE)
                if len(self._cache) > 512:            # keep memory bounded
                    for k in list(self._cache)[:256]:
                        if k != idx:
                            del self._cache[k]
            chunk = self._cache[idx]
            lo = idx * self.block
            out += chunk[max(0, start - lo): max(0, end - lo + 1)]
        return bytes(out)

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = self._fetch(self.pos, self.pos + n - 1)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "Hamner"
    outdir = sys.argv[2] if len(sys.argv) > 2 else None

    f = HTTPRangeFile(URL)
    print(f"remote zip: {f.size / 1e9:.1f} GB", flush=True)
    zf = zipfile.ZipFile(f)
    names = zf.namelist()
    print(f"total members: {len(names)}", flush=True)

    hits = [n for n in names if pattern.lower() in n.lower()]
    print(f"members matching {pattern!r}: {len(hits)}", flush=True)
    total = 0
    for info in (zf.getinfo(n) for n in hits):
        total += info.file_size
        print(f"  {info.filename}  {info.file_size / 1e6:.1f} MB", flush=True)
    print(f"matched payload: {total / 1e9:.3f} GB", flush=True)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        for n in hits:
            if n.endswith("/"):
                continue
            # keep the <subject>/<subject>.b3d layout MotionDataset walks over
            parts = n.rstrip("/").split("/")
            dest = os.path.join(outdir, parts[-2], parts[-1])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest) and os.path.getsize(dest) == zf.getinfo(n).file_size:
                print(f"skip (have) {dest}", flush=True)
                continue
            print(f"extracting {n} -> {dest}", flush=True)
            with zf.open(n) as src, open(dest, "wb") as dst:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    dst.write(buf)
        print("EXTRACT_DONE_OK", flush=True)


if __name__ == "__main__":
    main()
