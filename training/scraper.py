"""
scraper.py
==========
Scrapes images from DuckDuckGo for each custom class you want to train on.

Why DuckDuckGo?
---------------
- No API key required
- Generous limits for research / personal use
- Returns diverse, real-world images (not stock photos)

Usage
-----
python training/scraper.py --classes "earphones" "earbuds" "tws buds" --per-class 200
python training/scraper.py --classes "earphones" "smartwatch" --per-class 150 --output data/raw
"""

import argparse
import hashlib
import time
import urllib.request
from io import BytesIO
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional fast downloader — falls back gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from duckduckgo_search import DDGS          # pip install duckduckgo-search
    # optionally import exceptions if available
    try:
        from duckduckgo_search import exceptions as ddgs_exceptions
    except Exception:
        ddgs_exceptions = None
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

try:
    from PIL import Image                        # pip install Pillow
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

class ImageScraper:
    """
    Downloads images from DuckDuckGo image search.

    Parameters
    ----------
    output_dir : Path
        Root directory — images are saved to output_dir/<class_name>/
    min_size   : int
        Discard images smaller than this on either dimension (px).
    timeout    : int
        Per-image download timeout in seconds.
    """

    def __init__(
        self,
        output_dir: Path,
        min_size: int = 100,
        timeout: int = 8,
    ) -> None:
        if not HAS_DDGS:
            raise ImportError(
                "duckduckgo-search is not installed.\n"
                "Run:  pip install duckduckgo-search"
            )
        self.output_dir = Path(output_dir)
        self.min_size   = min_size
        self.timeout    = timeout

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def scrape(self, class_name: str, query: str, max_images: int) -> int:
        """
        Download up to *max_images* for *class_name* using *query*.

        Returns
        -------
        int  Number of images successfully saved.
        """
        save_dir = self.output_dir / _safe_name(class_name)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Scraper] '{class_name}'  query='{query}'  target={max_images}")

        # Resume support: detect already-saved images and their hashes
        seen: set[str] = set()   # dedup by content hash
        existing_files = sorted(save_dir.glob(f"{class_name}_*.jpg"))
        if existing_files:
            max_idx = -1
            for f in existing_files:
                try:
                    data = f.read_bytes()
                    seen.add(hashlib.md5(data).hexdigest())
                    # extract index from filename like <class>_0001.jpg
                    idx_part = f.stem.rsplit("_", 1)[-1]
                    idx = int(idx_part)
                    if idx > max_idx:
                        max_idx = idx
                except Exception:
                    continue
            saved = max_idx + 1
            print(f"[Scraper] Resuming '{class_name}' from {saved} existing images")
        else:
            saved = 0

        with DDGS() as ddgs:
            # Retry loop to handle transient rate limits from DuckDuckGo
            attempts = 0
            results = None
            while attempts < 10:
                try:
                    # reduce max_results to avoid extra DDG requests (helps with rate limits)
                    results = ddgs.images(
                        keywords=query,
                        max_results=max_images,   # fetch target amount (avoid over-querying)
                    )
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if 'ratelimit' in msg or '403' in msg or 'rate' in msg:
                        attempts += 1
                        wait = min(60, 5 * (2 ** (attempts-1)))
                        print(f"[Scraper] Rate-limited; sleeping {wait}s (attempt {attempts}/10)")
                        time.sleep(wait)
                        continue
                    raise
            if results is None:
                raise RuntimeError("Failed to fetch image results from DuckDuckGo after retries")

            for item in results:
                if saved >= max_images:
                    break
                url = item.get("image", "")
                if not url:
                    continue

                img_bytes = self._download(url)
                if img_bytes is None:
                    continue

                # Dedup check
                digest = hashlib.md5(img_bytes).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)

                # Validate / convert to JPEG
                img_path = save_dir / f"{class_name}_{saved:04d}.jpg"
                if not self._save_image(img_bytes, img_path):
                    continue

                saved += 1
                print(f"  [{saved:>4}/{max_images}] saved {img_path.name}", end="\r")

        print(f"\n[Scraper] '{class_name}' → {saved} images saved to {save_dir}")
        return saved

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download(self, url: str) -> bytes | None:
        """Download raw bytes from *url*; return None on any error."""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; research-scraper)"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except Exception:
            return None

    def _save_image(self, data: bytes, dest: Path) -> bool:
        """
        Validate image dimensions and save as JPEG.
        Returns True on success, False if the image should be skipped.
        """
        if not HAS_PIL:
            # Fallback: save raw bytes without validation
            dest.write_bytes(data)
            return True
        try:
            img = Image.open(BytesIO(data)).convert("RGB")
            w, h = img.size
            if w < self.min_size or h < self.min_size:
                return False   # too small — likely a thumbnail/icon
            img.save(dest, "JPEG", quality=90)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Convert a class name to a filesystem-safe directory name."""
    return name.lower().replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scrape training images from DuckDuckGo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape 200 images per class into data/raw/
  python training/scraper.py --classes "earphones" "earbuds" "tws buds" --per-class 200

  # Custom output directory
  python training/scraper.py --classes "smartwatch" "fitness band" --per-class 150 --output data/raw
        """,
    )
    p.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="One or more class names to scrape. "
             "Each name is used as the DuckDuckGo search query AND the YOLO class label.",
    )
    p.add_argument(
        "--per-class",
        type=int,
        default=200,
        dest="per_class",
        help="Target number of images per class. (default: 200)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Root directory to save scraped images. (default: data/raw)",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=80,
        dest="min_size",
        help="Minimum image dimension in pixels — smaller images are discarded. "
             "(default: 80)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    scraper = ImageScraper(
        output_dir=Path(args.output),
        min_size=args.min_size,
    )

    total = 0
    for cls in args.classes:
        # Use the class name as the search query; add "product photo" for better results
        query = f"{cls} product photo"
        n = scraper.scrape(cls, query=query, max_images=args.per_class)
        total += n
        time.sleep(1.5)   # polite delay between classes

    print(f"\n[Scraper] Done — {total} images across {len(args.classes)} classes.")
    print(f"[Scraper] Next step: run  python training/auto_label.py  to auto-label them.")


if __name__ == "__main__":
    main()