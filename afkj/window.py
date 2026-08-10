"""
ゲームウィンドウの検索・フォーカス・撮影・クリック
================================================
Win32 API を ctypes 経由で直接叩く（pygetwindow への依存をなくす）。
"""

from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes
from dataclasses import dataclass

import numpy as np
from PIL import ImageGrab

log = logging.getLogger(__name__)

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

# ─── DPI 対応（座標ズレ防止）────────────────────────────────────────────────
# import 時点で一度だけ実行する。ImageGrab / GetWindowRect の座標系を
# 物理ピクセルに揃えるために必須。


def _enable_dpi_awareness() -> str:
    """プロセスを DPI aware にする。設定できたモード名を返す。"""
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return "per-monitor"
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return "system"
    except Exception:
        return "none"


DPI_MODE = _enable_dpi_awareness()


# ─── ウィンドウ検索 ──────────────────────────────────────────────────────────

WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)


def _window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def find_window(title: str) -> int | None:
    """タイトルが部分一致する可視ウィンドウの hwnd を返す。

    同名ウィンドウが複数ある場合は面積が最大のものを選ぶ
    （ランチャーや通知の小窓を誤って掴まないため）。
    """
    found: list[tuple[int, int]] = []  # (面積, hwnd)

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        if title not in _window_title(hwnd):
            return True
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        area = (rect.right - rect.left) * (rect.bottom - rect.top)
        found.append((area, hwnd))
        return True

    user32.EnumWindows(WNDENUMPROC(callback), 0)
    if not found:
        return None
    found.sort(reverse=True)
    return found[0][1]


@dataclass(frozen=True)
class WindowRect:
    """ゲーム画面（クライアント領域）のスクリーン上の位置とサイズ。"""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        """クライアント領域内の座標 → スクリーン絶対座標"""
        return (self.left + x, self.top + y)


def get_client_rect(hwnd: int) -> WindowRect | None:
    """クライアント領域（枠を除いた描画領域）のスクリーン座標を返す。

    GetWindowRect ではなくクライアント領域を使うことで、
    タイトルバーや枠の有無に左右されずテンプレート座標が一致する。
    """
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None

    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    return WindowRect(origin.x, origin.y, width, height)


# ─── フォーカス ──────────────────────────────────────────────────────────────

SW_RESTORE = 9


def is_foreground(hwnd: int) -> bool:
    return user32.GetForegroundWindow() == hwnd


def is_minimized(hwnd: int) -> bool:
    return bool(user32.IsIconic(hwnd))


def focus_window(hwnd: int, timeout: float = 2.0) -> bool:
    """ウィンドウを前面に出す。すでに前面なら何もしない。

    Windows はフォアグラウンドの奪取を制限しているため、
    素直な SetForegroundWindow は失敗することがある。
    その場合は「入力スレッドを結びつける」定番の回避策を使う。
    """
    if is_foreground(hwnd) and not is_minimized(hwnd):
        return True

    # 長時間入力がないと SetForegroundWindow が無視される制限を解除
    SPI_SETFOREGROUNDLOCKTIMEOUT = 0x2001
    user32.SystemParametersInfoW(SPI_SETFOREGROUNDLOCKTIMEOUT, 0, None, 0)

    if is_minimized(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)

    user32.SetForegroundWindow(hwnd)

    if not _wait_foreground(hwnd, 0.5):
        # 回避策: 現在の前面ウィンドウのスレッドに自分を結びつけてから再試行
        fg = user32.GetForegroundWindow()
        cur_tid = ctypes.windll.kernel32.GetCurrentThreadId()
        fg_tid = user32.GetWindowThreadProcessId(fg, None)
        if fg_tid and fg_tid != cur_tid:
            user32.AttachThreadInput(cur_tid, fg_tid, True)
            try:
                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)
            finally:
                user32.AttachThreadInput(cur_tid, fg_tid, False)

    return _wait_foreground(hwnd, timeout)


def _wait_foreground(hwnd: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_foreground(hwnd):
            return True
        time.sleep(0.05)
    return is_foreground(hwnd)


# ─── 撮影 ────────────────────────────────────────────────────────────────────


def grab(rect: WindowRect) -> np.ndarray:
    """クライアント領域を撮影して RGB の numpy 配列で返す。

    注意: 画面に写っているものをそのまま撮る方式なので、
    ゲームが前面にある必要がある（Unity 製のため PrintWindow による
    背面撮影は使えないことを検証済み）。
    """
    img = ImageGrab.grab(bbox=rect.bbox, all_screens=True)
    return np.asarray(img.convert("RGB"))


# ─── クリック ────────────────────────────────────────────────────────────────

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("mi", MOUSEINPUT)]


def _mouse_input(nx: int, ny: int, flags: int) -> INPUT:
    return INPUT(
        type=INPUT_MOUSE,
        mi=MOUSEINPUT(dx=nx, dy=ny, mouseData=0, dwFlags=flags, time=0, dwExtraInfo=0),
    )


def click(screen_x: int, screen_y: int, hold: float = 0.06) -> None:
    """スクリーン絶対座標を SendInput でクリックする。

    マルチモニタでも正しく当たるよう仮想デスクトップ全体で正規化する。
    呼び出し側でゲームウィンドウを前面にしておくこと。
    """
    vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

    # SendInput の絶対座標は 0〜65535 に正規化した値を要求する
    nx = int(round((screen_x - vx) * 65535 / max(vw - 1, 1)))
    ny = int(round((screen_y - vy) * 65535 / max(vh - 1, 1)))
    flags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK

    events = [
        _mouse_input(nx, ny, flags | MOUSEEVENTF_MOVE),
        _mouse_input(nx, ny, flags | MOUSEEVENTF_LEFTDOWN),
        _mouse_input(nx, ny, flags | MOUSEEVENTF_LEFTUP),
    ]
    delays = [0.04, hold, 0.0]

    for event, delay in zip(events, delays):
        sent = user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
        if sent != 1:
            err = ctypes.get_last_error()
            log.warning("SendInput が拒否されました (err=%s)。管理者権限で実行していますか？", err)
        if delay:
            time.sleep(delay)


# ─── キー入力 ────────────────────────────────────────────────────────────────

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_ESCAPE = 0x1B


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYINPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("ki", KEYBDINPUT)]


def press_key(vk: int) -> None:
    """仮想キーコードを押して離す（想定外画面からの脱出用）。"""
    for flags in (0, KEYEVENTF_KEYUP):
        event = KEYINPUT(
            type=INPUT_KEYBOARD,
            ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0),
        )
        user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(KEYINPUT))
        time.sleep(0.04)


# ─── 停止操作の検出 ──────────────────────────────────────────────────────────

VK_F10 = 0x79


def is_key_down(vk: int) -> bool:
    """キーが今押されているか（フォーカスに関係なく調べられる）。"""
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def get_cursor_pos() -> tuple[int, int]:
    point = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return (point.x, point.y)


def failsafe_triggered(corner: int = 8) -> bool:
    """マウスが画面左上角にあるか（緊急停止のフェイルセーフ）。"""
    x, y = get_cursor_pos()
    vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    return x <= vx + corner and y <= vy + corner


# ─── 権限チェック ────────────────────────────────────────────────────────────


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


# ─── 高レベル API ────────────────────────────────────────────────────────────


class GameWindow:
    """ゲームウィンドウへの操作をまとめたハンドル。

    hwnd はゲームの再起動で変わるため、見失ったら再検索する。
    """

    def __init__(self, title: str):
        self.title = title
        self.hwnd: int | None = None

    def attach(self) -> bool:
        """ウィンドウを掴む（すでに有効ならそのまま）。"""
        if self.hwnd and user32.IsWindow(self.hwnd):
            return True
        self.hwnd = find_window(self.title)
        if self.hwnd:
            log.debug("ウィンドウを取得: hwnd=%s", self.hwnd)
            return True
        return False

    @property
    def rect(self) -> WindowRect | None:
        if not self.attach():
            return None
        return get_client_rect(self.hwnd)

    def focus(self) -> bool:
        if not self.attach():
            return False
        return focus_window(self.hwnd)

    def capture(self) -> tuple[np.ndarray, WindowRect] | None:
        """前面に出したうえで撮影する。失敗したら None。"""
        if not self.attach():
            return None
        if not self.focus():
            log.warning("ウィンドウを前面にできませんでした")
            return None
        rect = get_client_rect(self.hwnd)
        if rect is None:
            return None
        # 前面化直後は描画が追いつかないことがあるので少し待つ
        time.sleep(0.12)
        return grab(rect), rect

    def click_client(self, x: int, y: int) -> None:
        """クライアント領域内の座標をクリックする。"""
        rect = self.rect
        if rect is None:
            log.warning("ウィンドウが見つからないためクリックを中止")
            return
        sx, sy = rect.to_screen(x, y)
        self.focus()
        click(sx, sy)
