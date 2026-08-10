"""
画像照合（解像度非依存）
========================
テンプレート画像を切り出した時と実行時とで、ゲームの描画サイズが
違っていても当たるようにする。

なぜ必要か:
    - PC を変えると解像度もアスペクト比も変わる
    - 全画面 / ウィンドウの切り替えでも変わる
    - 同じ PC でもウィンドウを掴んでリサイズすれば変わる
  実測でも旧環境のスクショは 1588x1046〜1678x1054 とバラついていた。

やり方:
    テンプレートを複数の倍率に拡大縮小しながら照合し、最も一致した
    ものを採用する。まず縮小画像で粗く探して候補を絞り、良さそうな
    倍率の周辺だけ原寸で精査することで速度を保つ。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# 既定の一致しきい値（0〜1、高いほど厳密）
DEFAULT_THRESHOLD = 0.80

# 探索する倍率の範囲。画面サイズ比から求めた基準倍率に対する相対値。
# アスペクト比が変わると縦横で必要な倍率がずれるため、そのときは広めに探す。
SCALE_RATIOS_WIDE = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)
# アスペクト比が一致していれば倍率はほぼ一意に決まるので、狭く探せば足りる。
SCALE_RATIOS_NARROW = (0.96, 1.00, 1.04)
# アスペクト比が一致しているとみなす許容差
ASPECT_TOLERANCE = 0.02

# 照合を行う作業解像度の上限（高さ）。
# 対象がボタンや見出しといった大きな要素なので、縮めても精度は落ちない。
# 原寸のままだと1画面あたり3秒以上かかり実用にならなかった。
WORK_MAX_HEIGHT = 620

# 粗探索で使う縮小率（速度と精度の折り合い）
COARSE_SCALE = 0.5

# 最有力の倍率だけを先に原寸で試し、これ以上の一致なら即採用する。
# 実行時の画面サイズが切り出し時と同じなら倍率 1.0 でほぼ確実に当たるので、
# 通常はここで終わって全倍率の探索を省ける（実測で 1件あたり 0.7秒 → 0.1秒）。
EARLY_ACCEPT = 0.92


def imread_unicode(path: Path) -> np.ndarray | None:
    """日本語パスでも読める imread。

    cv2.imread は非ASCIIパスを開けないため np.fromfile を経由する。
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """日本語パスでも書ける imwrite。"""
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))
    return True


@dataclass
class Template:
    """1つのテンプレート画像とその照合条件。"""

    name: str
    path: Path
    ref_width: int  # 切り出し元スクショの横幅
    ref_height: int  # 切り出し元スクショの高さ
    threshold: float = DEFAULT_THRESHOLD
    note: str = ""

    _gray: np.ndarray | None = field(default=None, repr=False, compare=False)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def gray(self) -> np.ndarray | None:
        """グレースケール化したテンプレート（初回のみ読み込む）。"""
        if self._gray is None:
            img = imread_unicode(self.path)
            if img is None:
                log.error("テンプレート画像を読み込めません: %s", self.path)
                return None
            self._gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self._gray

    def scaled(self, scale: float) -> np.ndarray | None:
        """指定倍率にリサイズしたテンプレートを返す（結果をキャッシュ）。"""
        key = round(scale, 4)
        if key in self._cache:
            return self._cache[key]

        base = self.gray
        if base is None:
            return None
        h, w = base.shape[:2]
        nw, nh = max(int(round(w * scale)), 4), max(int(round(h * scale)), 4)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(base, (nw, nh), interpolation=interp)

        # キャッシュが際限なく増えないよう上限を設ける
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = resized
        return resized


class Frame:
    """1回分の画面。照合用の縮小画像を1度だけ作って使い回す。

    テンプレートごとに縮小し直すと同じ処理を何度も繰り返すことになるため、
    フレーム単位でまとめて持つ。
    """

    def __init__(self, gray: np.ndarray):
        self.gray = gray
        self.height, self.width = gray.shape[:2]
        factor = min(1.0, WORK_MAX_HEIGHT / max(self.height, 1))
        if factor < 1.0:
            self.work = cv2.resize(gray, None, fx=factor, fy=factor,
                                   interpolation=cv2.INTER_AREA)
        else:
            self.work = gray
        self.factor = factor
        self.work_height, self.work_width = self.work.shape[:2]

    def to_full(self, x: int, y: int) -> tuple[int, int]:
        """作業解像度の座標を、元の画面の座標へ戻す。"""
        if self.factor >= 1.0:
            return (x, y)
        return (int(round(x / self.factor)), int(round(y / self.factor)))


def as_frame(screen: "Frame | np.ndarray") -> Frame:
    """ndarray でも Frame でも受け取れるようにする。"""
    return screen if isinstance(screen, Frame) else Frame(screen)


@dataclass
class Match:
    """照合結果。座標はクライアント領域内のピクセル。"""

    name: str
    score: float
    center: tuple[int, int]
    box: tuple[int, int, int, int]  # (left, top, width, height)
    scale: float

    def __str__(self) -> str:
        return f"{self.name}(score={self.score:.3f}, at={self.center}, scale={self.scale:.2f})"


class TemplateStore:
    """templates.json とテンプレート画像を束ねて保持する。

    templates.json の形:
        {
          "templates": {
            "幻霊挑戦_btn": {
              "file": "幻霊挑戦_btn.png",
              "ref_size": [1920, 1200],
              "threshold": 0.80,
              "note": "ステージ選択画面 左下の緑ボタン"
            }
          }
        }
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.index_path = directory / "templates.json"
        self.templates: dict[str, Template] = {}

    # ── 読み書き ──────────────────────────────────────────────────────────

    def load(self) -> None:
        if not self.index_path.exists():
            log.warning("テンプレート定義がありません: %s", self.index_path)
            return
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.templates.clear()
        for name, spec in data.get("templates", {}).items():
            ref = spec.get("ref_size") or [0, 0]
            self.templates[name] = Template(
                name=name,
                path=self.directory / spec["file"],
                ref_width=int(ref[0]),
                ref_height=int(ref[1]),
                threshold=float(spec.get("threshold", DEFAULT_THRESHOLD)),
                note=spec.get("note", ""),
            )
        log.debug("テンプレートを %d件 読み込みました", len(self.templates))

    def save(self) -> None:
        data = {
            "templates": {
                name: {
                    "file": t.path.name,
                    "ref_size": [t.ref_width, t.ref_height],
                    "threshold": t.threshold,
                    "note": t.note,
                }
                for name, t in sorted(self.templates.items())
            }
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def add(self, name: str, file_name: str, ref_size: tuple[int, int], note: str = "") -> None:
        self.templates[name] = Template(
            name=name,
            path=self.directory / file_name,
            ref_width=ref_size[0],
            ref_height=ref_size[1],
            note=note,
        )

    def missing_files(self) -> list[str]:
        return [n for n, t in self.templates.items() if not t.path.exists()]

    # ── 照合 ──────────────────────────────────────────────────────────────

    def find(
        self, name: str, screen: "Frame | np.ndarray", threshold: float | None = None
    ) -> Match | None:
        """1つのテンプレートを画面から探す。見つからなければ None。"""
        template = self.templates.get(name)
        if template is None:
            log.warning("未定義のテンプレート: %s", name)
            return None
        return match_template(template, screen, threshold)

    def find_best(
        self, names: list[str], screen: "Frame | np.ndarray"
    ) -> Match | None:
        """複数のうち最も一致したものを返す。"""
        frame = as_frame(screen)
        best: Match | None = None
        for name in names:
            m = self.find(name, frame)
            if m and (best is None or m.score > best.score):
                best = m
        return best


def to_gray(screen_rgb: np.ndarray) -> np.ndarray:
    """撮影画像（RGB）を照合用のグレースケールにする。"""
    return cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2GRAY)


def _base_scales(template: Template, screen_h: int, screen_w: int) -> list[float]:
    """基準となる倍率の候補を作る。

    高さ比と横幅比の両方を基準に取る。アスペクト比が変わっている場合、
    UI が高さ基準で伸縮するか横幅基準かはゲーム次第なので両方試す。
    逆にアスペクト比が一致していれば倍率はほぼ一意なので、狭く探して速くする。
    """
    h_ratio = screen_h / template.ref_height if template.ref_height > 0 else None
    w_ratio = screen_w / template.ref_width if template.ref_width > 0 else None

    bases = [r for r in (h_ratio, w_ratio) if r]
    if not bases:
        return [1.0]

    same_aspect = (
        h_ratio is not None
        and w_ratio is not None
        and abs(h_ratio - w_ratio) / max(h_ratio, 1e-6) < ASPECT_TOLERANCE
    )
    ratios = SCALE_RATIOS_NARROW if same_aspect else SCALE_RATIOS_WIDE

    scales = {round(b * r, 4) for b in bases for r in ratios}
    return sorted(s for s in scales if s > 0.05)


def match_template(
    template: Template, screen: "Frame | np.ndarray", threshold: float | None = None
) -> Match | None:
    """マルチスケール照合。最良一致がしきい値以上なら Match を返す。

    照合は縮小した作業解像度の上で行い、座標だけ元の画面に戻す。
    2段階で探す:
      1. さらに縮小して全倍率をざっと試す（速い）
      2. 見込みのある倍率だけ作業解像度で精査する（正確）
    """
    frame = as_frame(screen)
    thr = template.threshold if threshold is None else threshold
    scales = _base_scales(template, frame.work_height, frame.work_width)

    # ── 0段階目: 最有力の倍率だけ先に試す ────────────────────────────
    # 画面サイズが切り出し時と同じなら、これで決まることがほとんど。
    # 外れたときだけ下の全倍率探索に進むので、精度は落とさずに速くできる。
    primary = _primary_scale(template, frame.work_height, frame.work_width)
    if primary is not None:
        quick = _match_at(template, frame, primary)
        if quick is not None and quick.score >= max(thr, EARLY_ACCEPT):
            return quick

    # ── 1段階目: 粗探索 ──────────────────────────────────────────────
    coarse_screen = cv2.resize(
        frame.work, None, fx=COARSE_SCALE, fy=COARSE_SCALE, interpolation=cv2.INTER_AREA
    )
    coarse_hits: list[tuple[float, float]] = []  # (score, scale)
    for scale in scales:
        tmpl = template.scaled(scale * COARSE_SCALE)
        if tmpl is None or not _fits(tmpl, coarse_screen):
            continue
        score = float(cv2.matchTemplate(coarse_screen, tmpl, cv2.TM_CCOEFF_NORMED).max())
        coarse_hits.append((score, scale))

    if not coarse_hits:
        log.debug("%s: どの倍率でも画面に収まりませんでした", template.name)
        return None

    # 粗探索で見込みのある倍率だけ残す（縮小の分だけ甘めに足切り）
    coarse_hits.sort(reverse=True)
    candidates = [s for score, s in coarse_hits[:2] if score >= thr - 0.22]
    if not candidates:
        return None

    # ── 2段階目: 作業解像度で精査 ────────────────────────────────────
    best: Match | None = None
    for scale in candidates:
        found = _match_at(template, frame, scale)
        if found is not None and (best is None or found.score > best.score):
            best = found

    if best is None:
        return None
    log.debug("%s: 最良 %.3f (しきい値 %.2f)", template.name, best.score, thr)
    return best if best.score >= thr else None


def _primary_scale(template: Template, screen_h: int, screen_w: int) -> float | None:
    """最も見込みの高い倍率。UI は高さ基準で伸縮することが多いのでそれを使う。"""
    if template.ref_height > 0:
        return screen_h / template.ref_height
    if template.ref_width > 0:
        return screen_w / template.ref_width
    return None


def _match_at(template: Template, frame: Frame, scale: float) -> Match | None:
    """指定倍率で1回だけ照合する。座標は元の画面のものに戻して返す。"""
    tmpl = template.scaled(scale)
    if tmpl is None or not _fits(tmpl, frame.work):
        return None
    result = cv2.matchTemplate(frame.work, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    h, w = tmpl.shape[:2]
    center = frame.to_full(max_loc[0] + w // 2, max_loc[1] + h // 2)
    left, top = frame.to_full(max_loc[0], max_loc[1])
    full_w, full_h = frame.to_full(w, h)
    return Match(
        name=template.name,
        score=float(max_val),
        center=center,
        box=(left, top, full_w, full_h),
        # 倍率は元画面を基準にした値で返す。内部の作業解像度は実装の都合なので、
        # ログや check の表示に出すと紛らわしい。
        scale=scale / frame.factor if frame.factor else scale,
    )


def _fits(template: np.ndarray, screen: np.ndarray) -> bool:
    """テンプレートが画面に収まるサイズか。"""
    th, tw = template.shape[:2]
    sh, sw = screen.shape[:2]
    return th <= sh and tw <= sw and th >= 4 and tw >= 4
