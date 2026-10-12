"""画面の数字を Windows 標準の OCR で読む
========================================
結社戦線の画面に出ている

    現在：22

のフロア番号を読む。

なぜ digits.py（数字の型との照合）を使わないのか:
    型を作るには 0〜9 がすべて写った画面が要るが、撮影できた画面の数字は
    19〜25 だけで 6・7・8 が無かった。字体も先鋒の集計画面とは違う。
    Windows には日本語の OCR エンジンが最初から入っているので、そちらで読む。
    実測: 撮影した結社戦線の画面で「現在：22」を 22 と読めた（等倍・3倍とも）。

PowerShell を1回起動するので 1回あたり 1.7秒ほどかかる。結社戦線の画面に
入ったときに1回読むだけなので、周回の速さには響かない。

読めなかったときは None を返す。誤った番号を記録するより、
読めなかったと残すほうがよい（digits.py と同じ方針）。
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .vision import imwrite_unicode

log = logging.getLogger(__name__)

SCRIPT = Path(__file__).with_name("ocr.ps1")

# 結社戦線の画面の「現在：22」（画面サイズに対する割合）。shot_032 で測った。
UNION_FLOOR_REGION = (880 / 1920, 805 / 1200, 1060 / 1920, 855 / 1200)

# PowerShell の起動を含めた待ち時間の上限（秒）
TIMEOUT = 20.0

# 小さな文字は拡大したほうが安定する
UPSCALE = 2.0


def read_text(image: np.ndarray) -> str | None:
    """画像（BGR または白黒）の文字を読む。失敗したら None。"""
    fd, name = tempfile.mkstemp(suffix=".png", prefix="afkj_ocr_")
    os.close(fd)
    path = Path(name)
    try:
        if not imwrite_unicode(path, image):
            log.debug("OCR 用の画像を保存できませんでした")
            return None
        proc = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-File", str(SCRIPT), "-Path", str(path),
            ],
            capture_output=True,
            timeout=TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("OCR を実行できませんでした: %s", exc)
        return None
    finally:
        path.unlink(missing_ok=True)

    if proc.returncode != 0:
        # PowerShell 自身のエラーはコンソールのコードページ（cp932）で出てくる
        err = proc.stderr.decode("cp932", errors="replace").strip()
        log.warning("OCR が失敗しました (終了コード %d): %s", proc.returncode, err[:200])
        return None
    return proc.stdout.decode("utf-8", errors="replace")


def crop_region(screen: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
    """割合で指定した範囲を切り出し、OCR 向けに拡大する。"""
    h, w = screen.shape[:2]
    x0, y0, x1, y1 = region
    crop = screen[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    return cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)


def parse_number(text: str) -> int | None:
    """OCR の結果から数字を1つ取り出す。

    日本語の OCR は文字の間に空白を入れる（「現 在 : 22」）ので、空白を
    詰めてから探す。数字の塊が2つ以上あったら、どれか決められないので読まない。
    """
    numbers = re.findall(r"\d+", re.sub(r"\s+", "", text))
    if len(numbers) != 1:
        return None
    return int(numbers[0])


def read_union_floor(screen: np.ndarray) -> int | None:
    """結社戦線の画面から「現在：N」の N を読む。"""
    text = read_text(crop_region(screen, UNION_FLOOR_REGION))
    if text is None:
        return None
    value = parse_number(text)
    if value is None:
        log.debug("OCR の結果から番号を取り出せませんでした: %r", text.strip())
    return value
