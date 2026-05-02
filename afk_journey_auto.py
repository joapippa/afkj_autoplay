"""
AFK Journey - 幻霊挑戦 自動操作スクリプト
================================================
動作環境: Windows / Python 3.8+

【必要ライブラリのインストール】
    pip install pyautogui opencv-python numpy pygetwindow pillow

【使い方】
    1. AFK Journey を起動し、「ステージ選択.png」の画面（幻霊挑戦ボタンが見える状態）にする
    2. このスクリプトを実行する
    3. 止めたい時は Ctrl+C または マウスを画面の左上角に移動（フェイルセーフ）

【動作フロー】
    Step1: 「幻霊挑戦」ボタンをクリック
    Step2: 「クリア編成」ボタンをクリック
    Step3: 「一括適用」ボタンをクリック
    Step4: 「オート挑戦」ボタンをクリック
    Step5: オート挑戦中... を待機
    Step6-1: 勝利 → 自動で次ステージへ → Step5へ
    Step6-2: 敗北 → 「オート戦闘終了」があればタップで閉じる → 「もう一度」→ Step2へ
"""

import os
import argparse
import ctypes
import pyautogui
import cv2
import numpy as np
import random
import time
import sys
import logging
from PIL import ImageGrab

# DPIスケーリング対応（座標ズレ防止）
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
try:
    import pygetwindow as gw
    HAS_PYGETWINDOW = True
except ImportError:
    HAS_PYGETWINDOW = False

try:
    import pydirectinput
    HAS_PYDIRECTINPUT = True
except ImportError:
    HAS_PYDIRECTINPUT = False

try:
    import win32api, win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

# ─── 設定 ───────────────────────────────────────────────────────────────────

# ゲームウィンドウタイトル（部分一致）
WINDOW_TITLE = "AFK Journey"

# テンプレート画像ファイルのパス（スクリプトと同じフォルダに置く）
# ★ 下記のパスを実際の画像ファイルのパスに変更してください
TEMPLATES = {
    "幻霊挑戦_btn":     "templates/幻霊挑戦_btn.png",       # Step1: 「幻霊挑戦」ボタン（左下）
    "挑戦_btn":         "templates/挑戦_btn.png",           # Step1: 「挑戦」ボタン（右下）
    "クリア編成_btn":   "templates/クリア編成_btn.png",     # Step2: 「クリア編成」ボタン
    "一括適用_btn":     "templates/一括適用_btn.png",       # Step3: 「一括適用」ボタン
    "オート挑戦_btn":   "templates/オート挑戦_btn.png",     # Step4: 「オート挑戦」ボタン
    "オート挑戦中":     "templates/オート挑戦中.png",       # Step5: 戦闘中の識別用
    "オート戦闘終了":   "templates/オート戦闘終了.png",     # Step6-2a: 「オート戦闘終了」画面
    "タップで閉じる":   "templates/タップで閉じる.png",     # Step6-2a: 閉じるボタン
    "戦闘敗北":         "templates/戦闘敗北.png",           # Step6-2: 敗北画面の識別用
    "もう一度_btn":     "templates/もう一度_btn.png",       # Step6-2: 「もう一度」ボタン
}

# 画像マッチング閾値（0.0〜1.0、高いほど厳密）
MATCH_THRESHOLD = 0.80

# 各操作後の待機時間（秒）
WAIT_AFTER_CLICK   = 1.0   # クリック後の待機
WAIT_BATTLE_CHECK  = 3.0   # 戦闘中のポーリング間隔
WAIT_SCREEN_TRANS  = 2.0   # 画面遷移待機
MAX_BATTLE_WAIT    = 3600  # 1戦闘の最大待機時間（秒）
MAX_RETRIES        = 50    # 最大リトライ回数（敗北ループ上限）

# ─── ログ設定 ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("afk_journey_auto.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ─── pyautogui 設定 ──────────────────────────────────────────────────────────

pyautogui.FAILSAFE = True   # マウスを左上角に移動で緊急停止
pyautogui.PAUSE    = 0.3    # 操作間の自動ウェイト


# ─── ユーティリティ関数 ──────────────────────────────────────────────────────

def get_game_window_rect():
    """ゲームウィンドウの位置とサイズを返す (left, top, right, bottom) or None"""
    if not HAS_PYGETWINDOW:
        return None
    wins = [w for w in gw.getAllWindows() if WINDOW_TITLE in w.title]
    if not wins:
        return None
    w = wins[0]
    return (w.left, w.top, w.right, w.bottom)


def screenshot_np() -> tuple:
    """
    ゲームウィンドウ領域のスクリーンショットをnumpy配列(BGR)で返す。
    Returns: (image, win_left, win_top)
             クリック座標補正用にウィンドウ左上座標も返す
    """
    rect = get_game_window_rect()
    if rect:
        left, top, right, bottom = rect
        img = ImageGrab.grab(bbox=(left, top, right, bottom))
        log.debug(f"ウィンドウ領域撮影: ({left},{top})-({right},{bottom})")
    else:
        log.warning("ウィンドウが見つからないためデスクトップ全体を撮影")
        img = ImageGrab.grab()
        left, top = 0, 0
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), left, top


def find_template(template_path: str, screen: np.ndarray,
                  threshold: float = MATCH_THRESHOLD):
    """
    テンプレートマッチング。
    Returns: (center_x, center_y) or None
    """
    # cv2.imread は日本語パスを読めないため np.fromfile で代替
    if not os.path.exists(template_path):
        log.warning(f"テンプレート画像が見つかりません: {template_path}")
        return None
    template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if template is None:
        log.warning(f"テンプレート画像の読み込みに失敗しました: {template_path}")
        return None

    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        h, w = template.shape[:2]
        cx = max_loc[0] + w // 2
        cy = max_loc[1] + h // 2
        log.debug(f"  マッチ: {template_path} ({max_val:.3f}) → ({cx}, {cy})")
        return (cx, cy)
    return None


def focus_game_window():
    """ゲームウィンドウをフォアグラウンドに持ってくる"""
    # Windowsのフォアグラウンドロックを無効化（長時間操作がないとSetForegroundWindowが失敗する）
    SPI_SETFOREGROUNDLOCKTIMEOUT = 0x2001
    ctypes.windll.user32.SystemParametersInfoW(SPI_SETFOREGROUNDLOCKTIMEOUT, 0, 0, 0)

    result = ctypes.c_ulong(0)
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong))

    def enum_callback(hwnd, lParam):
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            if WINDOW_TITLE in buf.value:
                lParam[0] = hwnd
                return False
        return True

    ctypes.windll.user32.EnumWindows(EnumWindowsProc(enum_callback), ctypes.byref(result))
    target_hwnd = result.value
    if target_hwnd:
        ctypes.windll.user32.ShowWindow(target_hwnd, 9)   # SW_RESTORE
        ctypes.windll.user32.SetForegroundWindow(target_hwnd)
        time.sleep(0.5)
        return True
    log.warning("ゲームウィンドウが見つかりません（フォーカス設定スキップ）")
    return False


def send_click(abs_x: int, abs_y: int):
    """SendInput を使って確実にクリックを送る（DPI対応済み座標を渡すこと）"""
    focus_game_window()

    INPUT_MOUSE = 0
    MOUSEEVENTF_MOVE        = 0x0001
    MOUSEEVENTF_LEFTDOWN    = 0x0002
    MOUSEEVENTF_LEFTUP      = 0x0004
    MOUSEEVENTF_ABSOLUTE    = 0x8000

    # 画面解像度取得（DPI対応）
    screen_w = ctypes.windll.user32.GetSystemMetrics(0)
    screen_h = ctypes.windll.user32.GetSystemMetrics(1)

    # SendInput が要求する正規化座標 (0〜65535)
    norm_x = int(abs_x * 65535 / screen_w)
    norm_y = int(abs_y * 65535 / screen_h)

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]

    move = INPUT(type=INPUT_MOUSE,
                 mi=MOUSEINPUT(dx=norm_x, dy=norm_y, mouseData=0,
                               dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
                               time=0, dwExtraInfo=None))
    down = INPUT(type=INPUT_MOUSE,
                 mi=MOUSEINPUT(dx=norm_x, dy=norm_y, mouseData=0,
                               dwFlags=MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE,
                               time=0, dwExtraInfo=None))
    up   = INPUT(type=INPUT_MOUSE,
                 mi=MOUSEINPUT(dx=norm_x, dy=norm_y, mouseData=0,
                               dwFlags=MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE,
                               time=0, dwExtraInfo=None))

    ctypes.windll.user32.SendInput(1, ctypes.byref(move), ctypes.sizeof(INPUT))
    time.sleep(0.05)
    ctypes.windll.user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(INPUT))
    time.sleep(0.05)
    ctypes.windll.user32.SendInput(1, ctypes.byref(up),   ctypes.sizeof(INPUT))


def click_template(key: str, screen_data: tuple,
                   wait: float = WAIT_AFTER_CLICK) -> bool:
    """
    テンプレートを探してクリック。
    Returns: クリックできたか
    """
    screen, win_left, win_top = screen_data
    pos = find_template(TEMPLATES[key], screen)
    if pos:
        # ウィンドウ内座標 → デスクトップ絶対座標に変換
        abs_x = pos[0] + win_left
        abs_y = pos[1] + win_top
        log.info(f"  クリック: {key} @ ウィンドウ内{pos} → 絶対({abs_x},{abs_y})")
        send_click(abs_x, abs_y)
        time.sleep(wait)
        return True
    return False


def save_debug_screenshot(label: str, screen_np: np.ndarray):
    """見つからないときの画面をdebug_screens/に保存する"""
    os.makedirs("debug_screens", exist_ok=True)
    ts = time.strftime("%H%M%S")
    # cv2.imwrite は日本語パスを書けないため imencode + tofile で代替
    path = f"debug_screens/{ts}_{label}.png"
    _, buf = cv2.imencode(".png", screen_np)
    buf.tofile(path)
    log.info(f"  デバッグ画像を保存: {path}")


def wait_for_template(key: str, timeout: float = 30.0,
                      interval: float = 1.0):
    """
    テンプレートが画面に現れるまで待つ。
    Returns: 見つかったら True、タイムアウトなら False
    """
    deadline = time.time() + timeout
    last_screen = None
    while time.time() < deadline:
        screen_data = screenshot_np()
        last_screen = screen_data[0]
        if find_template(TEMPLATES[key], last_screen):
            return True
        time.sleep(interval)
    # タイムアウト時にデバッグ画面を保存
    if last_screen is not None:
        save_debug_screenshot(f"timeout_{key}", last_screen)
    return False


def is_visible(key: str, screen_data: tuple) -> bool:
    return find_template(TEMPLATES[key], screen_data[0]) is not None


# ─── メインループ ────────────────────────────────────────────────────────────

def activate_game_window():
    """AFK Journey ウィンドウを前面に持ってくる"""
    if not HAS_PYGETWINDOW:
        log.warning("pygetwindow が使えないためウィンドウの自動アクティブ化をスキップ")
        return False
    wins = [w for w in gw.getAllWindows() if WINDOW_TITLE in w.title]
    if not wins:
        log.warning(f"ウィンドウ '{WINDOW_TITLE}' が見つかりません")
        return False
    win = wins[0]
    try:
        win.restore()
        win.activate()
        time.sleep(2.0)
        # クリック直前にウィンドウをクリックしてフォーカスを確実にする
        cx = win.left + win.width // 2
        cy = win.top + win.height // 2
        pyautogui.click(cx, cy)
        time.sleep(0.3)
        log.info(f"ウィンドウをアクティブ化: {win.title}")
        return True
    except Exception as e:
        log.warning(f"ウィンドウのアクティブ化に失敗: {e}")
        return False


def run(step1_mode: str):
    log.info("=" * 60)
    log.info("AFK Journey 幻霊挑戦 自動操作 開始")
    log.info(f"Step1モード: {step1_mode}")
    log.info("停止: Ctrl+C  または  マウスを画面左上角へ移動")
    log.info("=" * 60)

    retry_count = 0

    # ── ゲームウィンドウを自動でアクティブ化 ────────────────────────────
    log.info("  ゲームウィンドウをアクティブ化中...")
    if not focus_game_window():
        log.error(f"ゲームウィンドウ '{WINDOW_TITLE}' が見つかりません。ゲームを起動してください。")
        sys.exit(1)
    log.info("  操作開始！")

    # ── Step1: モードに応じたボタンをクリック ────────────────────────────
    if step1_mode == "random":
        target_key = random.choice(["幻霊挑戦_btn", "挑戦_btn"])
        log.info(f"[Step1] ランダム選択: {target_key}")
    elif step1_mode == "挑戦":
        target_key = "挑戦_btn"
    else:
        target_key = "幻霊挑戦_btn"

    log.info(f"[Step1] {target_key} を探しています...")
    screen_data = screenshot_np()
    if click_template(target_key, screen_data, wait=WAIT_SCREEN_TRANS):
        log.info(f"[Step1] 完了（{target_key}）→ 幻霊先鋒ステージ画面へ遷移中...")
    else:
        log.error(f"「{target_key}」が見つかりません。")
        log.error("ゲームを「ステージ選択」画面にしてから再実行してください。")
        sys.exit(1)

    # ── Step2以降のメインループ ───────────────────────────────────────────
    while retry_count < MAX_RETRIES:

        # Step2: クリア編成
        log.info(f"[Step2] クリア編成をクリック (試行 {retry_count + 1}回目)")
        if not wait_for_template("クリア編成_btn", timeout=15):
            log.error("「クリア編成」ボタンが見つかりません。処理を終了します。")
            sys.exit(1)
        screen_data = screenshot_np()
        click_template("クリア編成_btn", screen_data, wait=WAIT_SCREEN_TRANS)

        # Step3: 一括適用
        log.info("[Step3] 一括適用をクリック")
        if not wait_for_template("一括適用_btn", timeout=10):
            log.error("「一括適用」ボタンが見つかりません。処理を終了します。")
            sys.exit(1)
        screen_data = screenshot_np()
        click_template("一括適用_btn", screen_data, wait=WAIT_SCREEN_TRANS)

        # Step4: オート挑戦
        log.info("[Step4] オート挑戦をクリック")
        if not wait_for_template("オート挑戦_btn", timeout=10):
            log.error("「オート挑戦」ボタンが見つかりません。処理を終了します。")
            sys.exit(1)
        screen_data = screenshot_np()
        click_template("オート挑戦_btn", screen_data, wait=WAIT_SCREEN_TRANS)

        # Step5: 戦闘終了を待機
        log.info("[Step5] オート挑戦中... 戦闘終了を待機します")
        battle_start = time.time()
        battle_result = None

        while time.time() - battle_start < MAX_BATTLE_WAIT:
            time.sleep(WAIT_BATTLE_CHECK)
            screen_data = screenshot_np()

            # オート戦闘終了画面（勝利後の終了 or 敗北後の終了）
            if is_visible("オート戦闘終了", screen_data):
                log.info("[Step6-2a] 「オート戦闘終了」画面を検出")
                battle_result = "auto_end"
                break

            # 戦闘敗北画面（直接遷移のパターン）
            if is_visible("戦闘敗北", screen_data):
                log.info("[Step6-2] 「戦闘敗北」画面を検出（直接遷移）")
                battle_result = "defeat"
                break

            elapsed = int(time.time() - battle_start)
            log.info(f"  ... 戦闘中 ({elapsed}秒経過)")

        if battle_result is None:
            log.warning(f"戦闘が {MAX_BATTLE_WAIT}秒 以内に終わりませんでした。状態を確認してください。")
            sys.exit(1)

        # ── Step6 分岐 ──────────────────────────────────────────────────

        if battle_result == "auto_end":
            # オート戦闘終了画面 → 下部の「タップで閉じる」をクリック
            log.info("[Step6-2a] 「タップで閉じる」をクリック")
            screen_data = screenshot_np()
            if not click_template("タップで閉じる", screen_data, wait=WAIT_SCREEN_TRANS):
                # テンプレートが見つからない場合は画面下部（93%付近）をクリック
                log.warning("  「タップで閉じる」が見つからないため画面下部をクリック")
                rect = get_game_window_rect()
                if rect:
                    cx = (rect[0] + rect[2]) // 2
                    cy = rect[1] + int((rect[3] - rect[1]) * 0.93)
                else:
                    cx, cy = 960, 1000
                send_click(cx, cy)
                time.sleep(WAIT_SCREEN_TRANS)

            # 必ず戦闘敗北画面へ遷移する
            if not wait_for_template("戦闘敗北", timeout=10):
                log.error("「戦闘敗北」画面が見つかりません。処理を終了します。")
                sys.exit(1)
            screen_data = screenshot_np()
            battle_result = "defeat"

        if battle_result == "defeat":
            retry_count += 1
            log.info(f"[Step6-2] 「もう一度」をクリック → Step2へ戻ります (敗北 {retry_count}回)")
            screen_data = screenshot_np()
            if not click_template("もう一度_btn", screen_data, wait=WAIT_SCREEN_TRANS):
                log.error("「もう一度」ボタンが見つかりません。処理を終了します。")
                sys.exit(1)
            # Step2へ戻る（ループ継続）
            continue

    log.warning(f"最大リトライ回数 ({MAX_RETRIES}) に達しました。処理を終了します。")


# ─── エントリーポイント ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AFK Journey 幻霊挑戦 自動操作")
    parser.add_argument(
        "--step1",
        choices=["幻霊挑戦", "挑戦", "random"],
        default="幻霊挑戦",
        help="Step1で押すボタンを選択 (デフォルト: 幻霊挑戦)",
    )
    args = parser.parse_args()

    try:
        run(step1_mode=args.step1)
    except KeyboardInterrupt:
        log.info("\n[停止] ユーザーによる中断 (Ctrl+C)")
    except pyautogui.FailSafeException:
        log.info("\n[停止] フェイルセーフ発動 (マウスが左上角に移動されました)")
    except Exception as e:
        log.exception(f"予期しないエラー: {e}")
