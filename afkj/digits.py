"""集計画面の数字を読む
======================
オート戦闘終了（集計）画面の緑の帯に出る

    先鋒ステージ進捗   161 ≫ 165

の2つの数字を読み取る。差の 165-161=4 がその周回でクリアしたステージ数で、
これはゲーム自身が出している値なので、こちらで数える必要がない。

なぜ画像から読むのか:
    以前は戦闘画面の看板が変わった回数で数えていたが、今の版の戦闘画面には
    看板そのものが出ていない。あの領域に写っていたのは背景の街並みで、
    カメラが動くだけで数字が増えていた（実測: 0クリアの周回で3回計上、
    88クリアの周回で154回計上）。数えるのをやめ、ゲームの表示を読む。

読み方:
    数字は白（左）と緑（右）で塗りが違うが、輪郭の暗い縁取りは共通なので、
    暗い画素だけを取り出した2値の形で照合する。これで型は1組で足りる。
    桁は隣とくっついていて切り分けられないため、10個の型を領域全体に
    走査して当たりの強い順に採り、重なるものを捨てる。

    最後に「インクを覆えたか」を必ず確かめる。桁を1つ取り落とすと 91 が 1 に
    化けるが、覆えていない塊が残るので気づける。覆いきれなければ**読まない**。
    誤った数字を出すより、読めなかったと言うほうがよい。

実測（記録済みの集計画面58枚）:
    値が分かっている20枚 … 20枚とも正解、誤読 0
    残り38枚             … 36枚を読み取り、2枚は棄権（演出の粒子が数字に重なる）
    読めた56枚は前後がすべて連続していた（ある画面の左の数字＝前の画面の右の数字）
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .vision import imread_unicode

log = logging.getLogger(__name__)

# 数字を探す範囲（画面サイズに対する割合）。緑の帯の「進捗」の右側。
PROGRESS_REGION = (845 / 1920, 524 / 1200, 1220 / 1920, 574 / 1200)

# 縁取りとみなす暗さ。緑の帯・白い数字・緑の数字のいずれとも十分に離れている。
OUTLINE_DARKNESS = 95

# 一致とみなす強さ。まず STRONG で採り、覆えなかった隙間だけ WEAK で拾い直す。
STRONG_MATCH = 0.55
WEAK_MATCH = 0.30

# インクのうちこれだけ覆えていなければ、読み違いとみなして棄権する。
MIN_INK_COVERAGE = 0.85


class DigitReader:
    """数字の型（0〜9）を持ち、集計画面から2つの数字を読む。"""

    def __init__(self, directory: Path):
        self.directory = directory
        self.templates: dict[str, np.ndarray] = {}
        for ch in "0123456789":
            img = imread_unicode(directory / f"{ch}.png")
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.templates[ch] = (gray > 127).astype(np.float32)
        if not self.available:
            log.info("数字の型がありません (%s)。集計画面の数字は読みません", directory)

    @property
    def available(self) -> bool:
        return len(self.templates) == 10

    # ── 読み取り ──────────────────────────────────────────────────────────

    def read_progress(self, screen_gray: np.ndarray) -> tuple[int, int] | None:
        """(進捗の前, 進捗の後) を返す。自信が持てなければ None。"""
        if not self.available:
            return None

        mask = self._ink_mask(screen_gray)
        if mask is None:
            return None
        scale = screen_gray.shape[0] / 1200.0

        scores: dict[str, np.ndarray] = {}
        widths: dict[str, int] = {}
        for ch, tmpl in self.templates.items():
            sized = tmpl if abs(scale - 1.0) < 0.02 else cv2.resize(tmpl, None, fx=scale, fy=scale)
            if sized.shape[0] > mask.shape[0] or sized.shape[1] > mask.shape[1]:
                return None
            # 縦位置は決め打ちにせず、列ごとの最良だけを見る
            scores[ch] = cv2.matchTemplate(mask, sized, cv2.TM_CCOEFF_NORMED).max(axis=0)
            widths[ch] = sized.shape[1]

        found = self._collect(scores, widths)
        found += self._fill_holes(mask, found, scores, widths)
        if not found:
            return None

        found.sort(key=lambda f: f[1])
        if not self._covers_ink(mask, found):
            log.debug("集計画面の数字: 覆いきれないので棄権します")
            return None
        return self._split_two(found)

    # ── 内訳 ──────────────────────────────────────────────────────────────

    def _ink_mask(self, screen_gray: np.ndarray) -> np.ndarray | None:
        h, w = screen_gray.shape[:2]
        x1, y1, x2, y2 = PROGRESS_REGION
        crop = screen_gray[int(h * y1):int(h * y2), int(w * x1):int(w * x2)]
        if crop.size == 0:
            return None
        return (crop < OUTLINE_DARKNESS).astype(np.float32)

    def _collect(self, scores, widths) -> list[tuple[float, int, int, str]]:
        """強く当たったものから順に、重ならないように採っていく。"""
        cands = sorted(
            (
                (float(scores[ch][x]), int(x), widths[ch], ch)
                for ch in scores
                for x in range(len(scores[ch]))
                if scores[ch][x] >= STRONG_MATCH
            ),
            reverse=True,
        )
        taken: list[tuple[float, int, int, str]] = []
        for score, x, width, ch in cands:
            if _free(taken, x, width):
                taken.append((score, x, width, ch))
        return taken

    def _fill_holes(self, mask, taken, scores, widths) -> list[tuple[float, int, int, str]]:
        """どの桁にも覆われずに残ったインクを、緩い基準で拾い直す。

        隣とくっついた桁は当たりが弱くなって取りこぼしやすい。
        取りこぼすと 91 が 1 になるので、穴が空いていたらそこだけ見直す。
        """
        added: list[tuple[float, int, int, str]] = []
        for start, end in _runs(_ink_columns(mask) & ~_covered(mask, taken)):
            if end - start < 6:
                continue
            best = None
            for ch in scores:
                for x in range(max(0, start - 3), min(len(scores[ch]), end + 3)):
                    score = float(scores[ch][x])
                    if score < WEAK_MATCH:
                        continue
                    if best is not None and score <= best[0]:
                        continue
                    if _free(taken + added, x, widths[ch]):
                        best = (score, x, widths[ch], ch)
            if best is not None:
                added.append(best)
        return added

    def _covers_ink(self, mask, found) -> bool:
        ink = _ink_columns(mask)
        total = int(ink.sum())
        if total == 0:
            return False
        return int((ink & _covered(mask, found)).sum()) / total >= MIN_INK_COVERAGE

    def _split_two(self, found) -> tuple[int, int] | None:
        """並んだ桁を、いちばん広い空きで2つの数字に分ける（間にあるのは ≫）。"""
        gaps = [
            (found[i + 1][1] - (found[i][1] + found[i][2]), i) for i in range(len(found) - 1)
        ]
        if not gaps:
            return None
        _, at = max(gaps)
        left = "".join(f[3] for f in found[: at + 1])
        right = "".join(f[3] for f in found[at + 1:])
        if not left or not right:
            return None
        return int(left), int(right)


def _free(taken, x: int, width: int) -> bool:
    """すでに採った桁と重なっていないか。"""
    return all(x + width * 0.6 <= ox or ox + ow * 0.6 <= x for _, ox, ow, _ in taken)


def _ink_columns(mask: np.ndarray) -> np.ndarray:
    return mask.sum(axis=0) > 0


def _covered(mask: np.ndarray, found) -> np.ndarray:
    covered = np.zeros(mask.shape[1], bool)
    for _, x, width, _ in found:
        covered[x:x + width] = True
    return covered


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """True が続いている区間を (始まり, 終わり) で返す。"""
    out: list[tuple[int, int]] = []
    start = None
    for i, on in enumerate(flags):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out
