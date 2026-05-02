"""
テンプレート画像 切り出しツール（スクショファイル読み込み版）
=============================================================
事前に用意したスクリーンショットファイルを読み込んでテンプレートを切り出します。
ゲームとターミナルを切り替える必要はありません。

【フォルダ構成】
    afkj_autoplay/
    ├── make_templates.py   ← このスクリプト
    ├── screenshots/        ← スクショ6枚をここに置く
    │   ├── ステージ選択.png
    │   ├── 幻霊先鋒ステージ.png
    │   ├── クリア編成.png
    │   ├── オート挑戦中.png
    │   ├── オート戦闘終了.png
    │   └── 戦闘敗北.png
    └── templates/          ← 切り出した画像がここに保存される

【使い方】
    python make_templates.py

    ウィンドウが開くので切り出したい範囲をドラッグ → Enter で保存 / ESC でスキップ

【必要ライブラリ】
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import os
import sys

SCREENSHOTS_DIR = "screenshots"
OUTPUT_DIR = "templates"

# (テンプレートファイル名, 使用するスクショファイル名, 切り出す場所の説明)
TARGETS = [
    ("幻霊挑戦_btn",   "ステージ選択.png",       "左下の緑ボタン「幻霊挑戦」"),
    ("クリア編成_btn", "幻霊先鋒ステージ.png",   "左下の「クリア編成」ボタン"),
    ("オート挑戦_btn", "幻霊先鋒ステージ.png",   "下部中央の「オート挑戦」ボタン"),
    ("一括適用_btn",   "クリア編成.png",          "下部の緑ボタン「一括適用」"),
    ("オート挑戦中",   "オート挑戦中.png",        "画面下部「オート挑戦中...」テキスト"),
    ("オート戦闘終了", "オート戦闘終了.png",      "タイトル「オート戦闘終了」文字"),
    ("タップで閉じる", "オート戦闘終了.png",      "画面下部「タップで閉じる」テキスト"),
    ("戦闘敗北",       "戦闘敗北.png",            "タイトル「戦闘敗北」文字"),
    ("もう一度_btn",   "戦闘敗北.png",            "右下の緑ボタン「もう一度」"),
]

# ドラッグ操作用グローバル変数
_drag_start = None
_drag_end = None
_dragging = False


def mouse_callback(event, x, y, flags, param):
    global _drag_start, _drag_end, _dragging
    if event == cv2.EVENT_LBUTTONDOWN:
        _drag_start = (x, y)
        _drag_end = (x, y)
        _dragging = True
    elif event == cv2.EVENT_MOUSEMOVE and _dragging:
        _drag_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _drag_end = (x, y)
        _dragging = False


def get_rect():
    if _drag_start and _drag_end:
        x1 = min(_drag_start[0], _drag_end[0])
        y1 = min(_drag_start[1], _drag_end[1])
        x2 = max(_drag_start[0], _drag_end[0])
        y2 = max(_drag_start[1], _drag_end[1])
        return (x1, y1, x2, y2)
    return None


def show_and_crop(template_name, screenshot_path, description, screen_orig):
    """
    画像を表示してドラッグで範囲選択 → 切り出して保存。
    Returns: True=保存, False=スキップ
    """
    global _drag_start, _drag_end, _dragging
    _drag_start = None
    _drag_end = None
    _dragging = False

    orig_h, orig_w = screen_orig.shape[:2]

    # ウィンドウサイズを画面に収まるよう調整（最大1400x900）
    max_w, max_h = 1400, 900
    scale = min(max_w / orig_w, max_h / orig_h, 1.0)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)

    win_name = f"[{template_name}] {description}  |  ドラッグで選択 → Enter:保存  ESC:スキップ"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, disp_w, disp_h)
    cv2.setMouseCallback(win_name, mouse_callback)

    print(f"\n  {'─'*50}")
    print(f"  対象 : {template_name}")
    print(f"  説明 : {description}")
    print(f"  操作 : ドラッグで範囲選択 → Enter で保存 / ESC でスキップ")
    print(f"  {'─'*50}")

    while True:
        display = cv2.resize(screen_orig, (disp_w, disp_h))

        rect = get_rect()
        if rect:
            cv2.rectangle(display, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            # 選択サイズを表示
            real_w = int((rect[2] - rect[0]) / scale)
            real_h = int((rect[3] - rect[1]) / scale)
            label = f"{real_w}x{real_h}px"
            cv2.putText(display, label, (rect[0], max(rect[1]-8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif _dragging and _drag_start and _drag_end:
            cv2.rectangle(display, _drag_start, _drag_end, (0, 200, 255), 2)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # Enter
            rect = get_rect()
            if rect and (rect[2] - rect[0]) > 5 and (rect[3] - rect[1]) > 5:
                # スケールを元解像度に戻して切り出し
                x1 = int(rect[0] / scale)
                y1 = int(rect[1] / scale)
                x2 = int(rect[2] / scale)
                y2 = int(rect[3] / scale)
                cropped = screen_orig[y1:y2, x1:x2]
                out_path = os.path.join(OUTPUT_DIR, f"{template_name}.png")
                # cv2.imwrite は日本語パスを書けないため imencode + tofile で代替
                _, buf = cv2.imencode(".png", cropped)
                buf.tofile(out_path)
                print(f"  ✅ 保存: {out_path}  ({x2-x1}x{y2-y1}px)")
                cv2.destroyWindow(win_name)
                return True
            else:
                print("  ⚠  範囲が小さすぎます。もう一度ドラッグしてください。")

        elif key == 27:  # ESC
            print(f"  ⏭  スキップ: {template_name}")
            cv2.destroyWindow(win_name)
            return False


def main():
    print("=" * 60)
    print("  AFK Journey テンプレート画像 切り出しツール")
    print("  （スクショファイル読み込み版）")
    print("=" * 60)

    # フォルダ確認
    if not os.path.isdir(SCREENSHOTS_DIR):
        print(f"\n❌ スクショフォルダが見つかりません: {SCREENSHOTS_DIR}/")
        print(f"   スクリプトと同じ場所に '{SCREENSHOTS_DIR}' フォルダを作成し、")
        print(f"   スクショ6枚を入れてください。")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n出力先: {os.path.abspath(OUTPUT_DIR)}/\n")

    # スクショファイルのキャッシュ（同じファイルを複数回開かないよう）
    screenshot_cache = {}

    saved = []
    skipped = []
    errors = []

    for template_name, screenshot_file, description in TARGETS:
        out_path = os.path.join(OUTPUT_DIR, f"{template_name}.png")

        # すでに存在する場合は確認
        if os.path.exists(out_path):
            ans = input(f"\n  [{template_name}.png] はすでに存在します。上書きしますか？ [y/N]: ").strip().lower()
            if ans != "y":
                print(f"  ⏭  スキップ: {template_name}")
                skipped.append(template_name)
                continue

        # スクショ読み込み（キャッシュ活用）
        screenshot_path = os.path.join(SCREENSHOTS_DIR, screenshot_file)
        if screenshot_file not in screenshot_cache:
            if not os.path.exists(screenshot_path):
                print(f"\n  ❌ スクショが見つかりません: {screenshot_path}")
                print(f"     '{SCREENSHOTS_DIR}/' フォルダに '{screenshot_file}' を置いてください。")
                errors.append(template_name)
                continue
            # cv2.imread は日本語パスを読めないため np.fromfile で代替
            img = cv2.imdecode(
                np.fromfile(screenshot_path, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is None:
                print(f"\n  ❌ 画像の読み込みに失敗しました: {screenshot_path}")
                errors.append(template_name)
                continue
            screenshot_cache[screenshot_file] = img
            print(f"\n  📂 読み込み: {screenshot_file}  ({img.shape[1]}x{img.shape[0]}px)")

        screen = screenshot_cache[screenshot_file]
        result = show_and_crop(template_name, screenshot_path, description, screen)

        if result:
            saved.append(template_name)
        else:
            skipped.append(template_name)

    # 結果サマリー
    print("\n" + "=" * 60)
    print(f"  完了: {len(saved)}件保存 / {len(skipped)}件スキップ / {len(errors)}件エラー")
    if saved:
        print(f"  保存: {', '.join(saved)}")
    if skipped:
        print(f"  スキップ: {', '.join(skipped)}")
    if errors:
        print(f"  エラー: {', '.join(errors)}")
    print("=" * 60)

    total_done = len(saved) + len(skipped)
    total_needed = len(TARGETS) - len(errors)
    if total_done == len(TARGETS) and not errors:
        print("\n✅ 全テンプレート準備完了！次のコマンドで自動操作を開始できます：")
        print("   python afk_journey_auto.py")
    elif errors:
        print(f"\n⚠  スクショファイルが不足しています。'{SCREENSHOTS_DIR}/' を確認してください。")


if __name__ == "__main__":
    main()
