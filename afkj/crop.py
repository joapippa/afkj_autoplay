"""
テンプレート切り出しツール
==========================
撮影したスクショから、照合に使う部分（ボタンや見出し）を切り出す。

切り出し元スクショの解像度を templates.json に記録するのが要点。
実行時の画面サイズとの比から倍率を割り出すため、これがないと
解像度が変わったときに当たらなくなる。

操作:
    ドラッグ      範囲を選ぶ
    Enter        選んだ範囲を保存（名前はターミナルで入力）
    N / →        次のスクショへ
    P / ←        前のスクショへ
    R            選択をやり直す
    Q / ESC      終了
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .vision import TemplateStore, imread_unicode, imwrite_unicode

log = logging.getLogger(__name__)

# 表示ウィンドウの最大サイズ（画面からはみ出さないように）
MAX_VIEW_W, MAX_VIEW_H = 1500, 850

WIN_NAME = "crop"


class _Selection:
    """ドラッグ中の矩形選択を保持する。"""

    def __init__(self) -> None:
        self.start: tuple[int, int] | None = None
        self.end: tuple[int, int] | None = None
        self.dragging = False

    def reset(self) -> None:
        self.start = self.end = None
        self.dragging = False

    def rect(self) -> tuple[int, int, int, int] | None:
        """(x1, y1, x2, y2) を返す。未選択なら None。"""
        if not self.start or not self.end:
            return None
        x1, x2 = sorted((self.start[0], self.end[0]))
        y1, y2 = sorted((self.start[1], self.end[1]))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return (x1, y1, x2, y2)

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.end = (x, y)
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end = (x, y)
            self.dragging = False


def run(screenshots_dir: Path, templates_dir: Path) -> int:
    """切り出しツールのメインループ。保存した数を返す。"""
    shots = sorted(p for p in screenshots_dir.glob("*.png"))
    if not shots:
        print(f"[エラー] スクショが1枚もありません: {screenshots_dir.resolve()}")
        print("        先に `afkj capture` で撮影してください。")
        return 0

    store = TemplateStore(templates_dir)
    store.load()

    print("=" * 64)
    print("  テンプレート切り出しツール")
    print("=" * 64)
    print(f"  スクショ   : {len(shots)}枚  ({screenshots_dir.resolve()})")
    print(f"  保存先     : {templates_dir.resolve()}")
    print(f"  登録済み   : {len(store.templates)}件")
    print()
    print("  ドラッグで範囲を選び、Enter で保存します。")
    print("    N: 次のスクショ   P: 前のスクショ   R: 選択やり直し   Q: 終了")
    print("=" * 64)

    sel = _Selection()
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_NAME, sel.on_mouse)

    index = 0
    saved = 0

    try:
        while 0 <= index < len(shots):
            path = shots[index]
            image = imread_unicode(path)
            if image is None:
                print(f"  [警告] 読み込めません: {path.name}")
                index += 1
                continue

            src_h, src_w = image.shape[:2]
            scale = min(MAX_VIEW_W / src_w, MAX_VIEW_H / src_h, 1.0)
            view_w, view_h = int(src_w * scale), int(src_h * scale)
            view_base = cv2.resize(image, (view_w, view_h), interpolation=cv2.INTER_AREA)

            cv2.resizeWindow(WIN_NAME, view_w, view_h)
            sel.reset()
            print(f"\n  [{index + 1}/{len(shots)}] {path.name}  ({src_w}x{src_h})")

            action = _edit_one(sel, view_base, scale, path, src_w, src_h, store, templates_dir)

            if action == "quit":
                break
            if action == "next":
                index += 1
            elif action == "prev":
                index = max(index - 1, 0)
            elif isinstance(action, int):
                saved += action
                # 同じスクショから続けて切り出せるよう index は動かさない
    finally:
        cv2.destroyAllWindows()
        if saved:
            store.save()
            print(f"\n  templates.json を更新しました ({len(store.templates)}件)")

    print("\n" + "=" * 64)
    print(f"  切り出し終了: {saved}件 保存しました")
    print("=" * 64)
    return saved


def _edit_one(
    sel: _Selection,
    view_base: np.ndarray,
    scale: float,
    path: Path,
    src_w: int,
    src_h: int,
    store: TemplateStore,
    templates_dir: Path,
):
    """1枚のスクショに対する操作ループ。

    戻り値: "quit" / "next" / "prev" / 保存した件数(int)
    """
    saved_here = 0

    while True:
        view = view_base.copy()
        rect = sel.rect()
        if rect:
            x1, y1, x2, y2 = rect
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            real_w = int((x2 - x1) / scale)
            real_h = int((y2 - y1) / scale)
            cv2.putText(
                view, f"{real_w}x{real_h}px", (x1, max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
        elif sel.dragging and sel.start and sel.end:
            cv2.rectangle(view, sel.start, sel.end, (0, 200, 255), 2)

        cv2.imshow(WIN_NAME, view)
        key = cv2.waitKey(20) & 0xFF

        # ウィンドウの × で閉じられた場合も終了扱いにする
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return "quit"

        if key in (ord("q"), 27):
            return "quit"
        if key in (ord("n"), 83):  # 83 = →
            return saved_here or "next"
        if key in (ord("p"), 81):  # 81 = ←
            return "prev"
        if key == ord("r"):
            sel.reset()
            continue

        if key == 13:  # Enter
            rect = sel.rect()
            if not rect:
                print("     範囲が選ばれていません。ドラッグしてください。")
                continue
            if _save_crop(rect, scale, path, src_w, src_h, store, templates_dir):
                saved_here += 1
                sel.reset()


def _save_crop(
    rect: tuple[int, int, int, int],
    scale: float,
    path: Path,
    src_w: int,
    src_h: int,
    store: TemplateStore,
    templates_dir: Path,
) -> bool:
    """選択範囲を切り出して保存し、templates.json に登録する。"""
    name = input("     テンプレート名（空Enterで取消）: ").strip()
    if not name:
        print("     取消しました。")
        return False

    if name in store.templates:
        ans = input(f"     '{name}' は登録済みです。上書きしますか？ [y/N]: ").strip().lower()
        if ans != "y":
            print("     取消しました。")
            return False

    # 表示上の座標を元画像の座標へ戻す
    x1, y1, x2, y2 = (int(v / scale) for v in rect)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, src_w), min(y2, src_h)

    source = imread_unicode(path)
    if source is None:
        print("     [エラー] 元画像を読み込めませんでした。")
        return False

    cropped = source[y1:y2, x1:x2]
    if cropped.size == 0:
        print("     [エラー] 範囲が不正です。")
        return False

    out_path = templates_dir / f"{name}.png"
    if not imwrite_unicode(out_path, cropped):
        print("     [エラー] 保存に失敗しました。")
        return False

    store.add(
        name=name,
        file_name=out_path.name,
        ref_size=(src_w, src_h),
        note=f"{path.name} から切り出し",
    )
    store.save()
    print(f"     保存: {out_path.name}  ({x2 - x1}x{y2 - y1}px / 元画面 {src_w}x{src_h})")
    return True
