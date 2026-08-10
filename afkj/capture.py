"""
撮影モード
==========
ゲームをプレイしている間に、画面が切り替わったタイミングを自動で見つけて
保存する。キーを押す必要はない。

    自動保存   画面が切り替わって落ち着いたら1枚保存する
    F9        今すぐ1枚保存する（自動保存と関係なく撮りたいとき）
    F10       終了
    Ctrl+C    終了

実装上の注意:
    ホットキーの登録（RegisterHotKey）とメッセージループは使わない。
    GetMessage は C 側でブロックするため、その間 Python が Ctrl+C を
    処理できずターミナルごと固まってしまう。ここでは GetAsyncKeyState を
    短い間隔で見に行く方式にしている。登録が要らないので他のアプリと
    ホットキーが衝突することもない。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import window as win

log = logging.getLogger(__name__)

VK_F9 = 0x78
VK_F10 = 0x79

# 画面の変化量の判定（0〜1。32x32 に縮めた輝度の平均差）
SETTLED_DIFF = 0.012  # これ未満なら「画面が動いていない＝遷移が終わった」
NEW_SCREEN_DIFF = 0.045  # これを超えたら「別の画面になった」

# 完全には静止しないが、大きくは動いていない状態の上限。
# 編成画面のようにキャラクターが常に動いている画面は、静止条件だけだと
# いつまでも保存されない。少し動いている程度ならこの猶予で拾う。
ALMOST_SETTLED_DIFF = 0.05
# 別画面のまま静止しない状態がこの秒数続いたら、静止を待たずに保存する
FORCE_SAVE_AFTER = 2.5

# 自動保存の最短間隔（秒）。戦闘中の演出で撮りすぎないための保険。
# 短くしすぎると似た画面が増えるが、長すぎると一瞬しか出ない画面
# （編成画面など、すぐ次へ進んでしまうもの）を取りこぼす。
MIN_SAVE_INTERVAL = 0.8


def _signature(rgb: np.ndarray) -> np.ndarray:
    """画面を比較するための小さな指紋を作る。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def _diff(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 1.0
    return float(np.abs(a - b).mean())


def _next_index(out_dir: Path, prefix: str) -> int:
    """既存ファイルを見て次の連番を決める（上書き事故を防ぐ）。"""
    max_n = 0
    for path in out_dir.glob(f"{prefix}_*.png"):
        stem = path.stem[len(prefix) + 1 :]
        if stem.isdigit():
            max_n = max(max_n, int(stem))
    return max_n + 1


def run(
    window_title: str,
    out_dir: Path,
    prefix: str = "shot",
    auto: bool = True,
    interval: float = 0.7,
) -> int:
    """撮影モードのメインループ。保存した枚数を返す。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    game = win.GameWindow(window_title)

    if not game.attach():
        print(f"[エラー] ウィンドウ '{window_title}' が見つかりません。ゲームを起動してください。")
        return 0

    rect = game.rect
    print("=" * 64)
    print("  撮影モード")
    print("=" * 64)
    print(f"  対象ウィンドウ : {window_title}  ({rect.width}x{rect.height})")
    print(f"  保存先         : {out_dir.resolve()}")
    print(f"  自動保存       : {'ON（画面が切り替わったら自動で撮ります）' if auto else 'OFF'}")
    print()
    print("  そのままゲームをプレイしてください。ターミナルに戻る必要はありません。")
    print("    F9      今すぐ1枚撮る")
    print("    F10     終了")
    print("    Ctrl+C  終了")
    print()
    print("  ※ ゲームが前面にある間だけ撮影します。別の作業に切り替えている間は")
    print("     何もしないので、放っておいて大丈夫です。")
    print("=" * 64)
    print()

    count = 0
    index = _next_index(out_dir, prefix)

    prev_sig: np.ndarray | None = None  # 直前フレーム（画面が落ち着いたかの判定用）
    saved_sig: np.ndarray | None = None  # 最後に保存したフレーム
    changed_since: float | None = None  # 別画面になったのに未保存の状態が続いた開始時刻
    last_save_at = 0.0
    f9_was_down = False
    started = time.time()
    last_status = 0.0

    try:
        while True:
            # ── 終了操作 ──────────────────────────────────────────────
            if win.is_key_down(VK_F10):
                print("\n  F10 が押されました。終了します。")
                break

            f9_down = win.is_key_down(VK_F9)
            f9_pressed = f9_down and not f9_was_down  # 押した瞬間だけ拾う
            f9_was_down = f9_down

            # ── ゲームが前面にあるときだけ見る ────────────────────────
            if not game.attach() or not win.is_foreground(game.hwnd):
                prev_sig = None
                _status(started, count, "ゲームが前面にありません（待機中）", last_status)
                last_status = time.time()
                time.sleep(0.4)
                continue

            rect = win.get_client_rect(game.hwnd)
            if rect is None:
                time.sleep(0.4)
                continue

            rgb = win.grab(rect)
            sig = _signature(rgb)

            # ── 保存するか判断する ────────────────────────────────────
            reason = None
            if f9_pressed:
                reason = "F9"
            elif auto:
                motion = _diff(prev_sig, sig)
                settled = motion < SETTLED_DIFF
                changed = _diff(saved_sig, sig) > NEW_SCREEN_DIFF
                long_enough = time.time() - last_save_at >= MIN_SAVE_INTERVAL

                # 別の画面になってから、まだ保存できずにいる時間を測る
                if changed:
                    if changed_since is None:
                        changed_since = time.time()
                else:
                    changed_since = None

                # 「別の画面になった」かつ「もう動いていない」ときに撮る。
                # 遷移アニメの途中や戦闘中の演出で撮りすぎないための条件。
                if settled and changed and long_enough:
                    reason = "自動"
                # ただし静止を待つだけだと、キャラクターが動き続ける画面
                # （編成画面など）が永久に撮れない。大きく動いていなければ
                # 一定時間後に静止を待たずに撮る。
                elif (
                    changed
                    and long_enough
                    and motion < ALMOST_SETTLED_DIFF
                    and changed_since is not None
                    and time.time() - changed_since >= FORCE_SAVE_AFTER
                ):
                    reason = "自動(動きあり)"

            if reason:
                changed_since = None
                path = out_dir / f"{prefix}_{index:03d}.png"
                Image.fromarray(rgb).save(path)
                count += 1
                index += 1
                saved_sig = sig
                last_save_at = time.time()
                print(f"  [{reason}] 保存: {path.name}  ({rect.width}x{rect.height})  "
                      f"合計 {count}枚          ")

            prev_sig = sig
            _status(started, count, "撮影中", last_status)
            last_status = time.time()
            time.sleep(interval if not auto else min(interval, 0.7))

    except KeyboardInterrupt:
        print("\n  Ctrl+C で終了します。")

    print()
    print("=" * 64)
    print(f"  撮影終了: {count}枚 保存しました → {out_dir.resolve()}")
    print("=" * 64)
    return count


def _status(started: float, count: int, note: str, last_status: float) -> None:
    """動いていることが分かるよう1行だけ更新し続ける。"""
    if time.time() - last_status < 1.0:
        return
    elapsed = int(time.time() - started)
    mins, secs = divmod(elapsed, 60)
    sys.stdout.write(f"\r  [{mins:02d}:{secs:02d}] {note}  保存 {count}枚   ")
    sys.stdout.flush()
