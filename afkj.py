"""
AFK Journey 自動操作ツール
==========================
    python afkj.py doctor     環境を診断する（最初にこれを実行）
    python afkj.py capture    ゲーム画面を撮影する（F9で保存 / F10で終了）
    python afkj.py crop       撮ったスクショからテンプレート画像を切り出す
    python afkj.py run        自動操作を開始する

必要ライブラリ:
    pip install opencv-python numpy pillow
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCREENSHOTS_DIR = ROOT / "screenshots"
TEMPLATES_DIR = ROOT / "templates"
DEBUG_DIR = ROOT / "debug_screens"

WINDOW_TITLE = "AFK Journey"


def setup_console_utf8() -> None:
    """コンソールを UTF-8 にする。

    日本語ログが cmd / PowerShell / Git Bash のどれでも化けないようにするため。
    コンソールの出力コードページと Python 側の encoding の両方を合わせる必要がある。
    """
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "afkj.log", encoding="utf-8"),
        ],
    )


# ─── doctor ──────────────────────────────────────────────────────────────────


def cmd_doctor(args: argparse.Namespace) -> int:
    print("=" * 64)
    print("  環境診断")
    print("=" * 64)

    ok = True

    print(f"\n[Python]")
    print(f"  バージョン : {sys.version.split()[0]}")
    print(f"  実行ファイル: {sys.executable}")

    print(f"\n[ライブラリ]")
    for mod, pkg in [("cv2", "opencv-python"), ("numpy", "numpy"), ("PIL", "pillow")]:
        try:
            __import__(mod)
            print(f"  OK   {pkg}")
        except ImportError:
            print(f"  未   {pkg}  → pip install {pkg}")
            ok = False

    if not ok:
        print("\n必要なライブラリが不足しています。上記のコマンドでインストールしてください。")
        return 1

    from afkj import window as win

    print(f"\n[Windows]")
    print(f"  DPI対応   : {win.DPI_MODE}")
    admin = win.is_admin()
    print(f"  管理者権限 : {'あり' if admin else 'なし'}")
    if not admin:
        print("       ⚠ 管理者権限がないとクリックがゲームに届かない場合があります。")
        print("         ターミナルを右クリック →「管理者として実行」で開き直してください。")

    print(f"\n[ゲームウィンドウ]")
    game = win.GameWindow(WINDOW_TITLE)
    if not game.attach():
        print(f"  見つかりません: '{WINDOW_TITLE}'")
        print("       ゲームを起動してから、もう一度実行してください。")
        ok = False
    else:
        rect = game.rect
        print(f"  hwnd      : {game.hwnd}")
        print(f"  描画領域   : {rect.width} x {rect.height}  (位置 {rect.left},{rect.top})")
        print(f"  前面表示   : {'はい' if win.is_foreground(game.hwnd) else 'いいえ'}")
        print(f"  最小化     : {'はい' if win.is_minimized(game.hwnd) else 'いいえ'}")

    print(f"\n[ファイル]")
    shots = sorted(SCREENSHOTS_DIR.glob("*.png")) if SCREENSHOTS_DIR.is_dir() else []
    templates = sorted(TEMPLATES_DIR.glob("*.png")) if TEMPLATES_DIR.is_dir() else []
    print(f"  screenshots/ : {len(shots)}枚")
    print(f"  templates/   : {len(templates)}枚")
    if templates:
        for t in templates:
            print(f"      - {t.name}")

    print("\n" + "=" * 64)
    if not ok:
        print("  上の ⚠ / 未 の項目を解消してから、もう一度実行してください。")
    elif not (TEMPLATES_DIR / "templates.json").exists():
        print("  診断OK。次は `afkj capture` で画面を撮影してください。")
    else:
        print("  診断OK。準備はできています。")
        print("    afkj check            今の画面の一致状況を見る")
        print("    afkj run --dry-run    判定だけ試す（クリックしない）")
        print("    afkj run              自動操作を開始する")
    print("=" * 64)
    return 0 if ok else 1


# ─── capture ─────────────────────────────────────────────────────────────────


def cmd_capture(args: argparse.Namespace) -> int:
    from afkj import capture

    out_dir = Path(args.out) if args.out else SCREENSHOTS_DIR
    count = capture.run(
        WINDOW_TITLE,
        out_dir,
        prefix=args.prefix,
        auto=not args.manual,
        interval=args.interval,
    )
    return 0 if count else 1


# ─── crop ────────────────────────────────────────────────────────────────────


def cmd_crop(args: argparse.Namespace) -> int:
    from afkj import crop

    src = Path(args.src) if args.src else SCREENSHOTS_DIR
    saved = crop.run(src, TEMPLATES_DIR)
    return 0 if saved else 1


# ─── check（テンプレートの効きを確認）───────────────────────────────────────


def cmd_check(args: argparse.Namespace) -> int:
    """今のゲーム画面に対して、各テンプレートがどれだけ一致するかを一覧表示する。

    ゲームの更新でどのボタンが変わったかを切り分けるのに使う。
    クリックは一切しない。
    """
    from afkj import window as win
    from afkj.vision import Frame, TemplateStore, to_gray

    store = TemplateStore(TEMPLATES_DIR)
    store.load()
    if not store.templates:
        print(f"[エラー] テンプレートが登録されていません: {store.index_path}")
        print("        `afkj capture` → `afkj crop` の順に実行してください。")
        return 1

    missing = store.missing_files()
    if missing:
        print(f"[警告] 画像ファイルが見つからないテンプレート: {', '.join(missing)}")

    game = win.GameWindow(WINDOW_TITLE)
    shot = game.capture()
    if shot is None:
        print(f"[エラー] ゲームウィンドウ '{WINDOW_TITLE}' を撮影できませんでした。")
        print("        ゲームを起動して、最小化していない状態にしてください。")
        return 1

    rgb, rect = shot
    gray = Frame(to_gray(rgb))

    print("=" * 64)
    print(f"  テンプレート一致チェック   画面 {rect.width}x{rect.height}")
    print("=" * 64)
    print(f"  {'テンプレート':<20}{'一致':>8}{'倍率':>8}   位置")
    print("  " + "-" * 60)

    hits = 0
    matched_names: set[str] = set()
    for name in sorted(store.templates):
        match = store.find(name, gray, threshold=0.0)
        if match is None:
            print(f"  {name:<20}{'-':>8}{'-':>8}   （照合できず）")
            continue
        thr = store.templates[name].threshold
        ok = match.score >= thr
        hits += ok
        if ok:
            matched_names.add(name)
        mark = "OK " if ok else ("~  " if match.score >= thr - 0.12 else "NG ")
        print(f"  {name:<20}{mark}{match.score:>5.3f}{match.scale:>8.2f}   {match.center}")

    print("  " + "-" * 60)
    print(f"  一致: {hits} / {len(store.templates)}")
    print("=" * 64)

    _warn_contradictions(matched_names)

    if hits < len(store.templates):
        print("  NG のものは、今の画面に写っていないだけかもしれません。")
        print("  各画面で実行して切り分けてください。どの画面でも NG なら作り直しです。")
    return 0


def _warn_contradictions(matched: set[str]) -> None:
    """同時に成立しないはずの画面が両方反応していないか調べる。"""
    from afkj import states as st

    for group in st.EXCLUSIVE_GROUPS:
        reacted = [
            st.STATES_BY_KEY[key].label
            for key in group
            if any(name in matched for name in st.STATES_BY_KEY[key].detect)
        ]
        if len(reacted) >= 2:
            print()
            print(f"  ⚠ {' と '.join(reacted)} が同時に反応しています。")
            print("    これらは同時には起こらないので、テンプレートが2画面を")
            print("    見分けられていません。片方だけが反応するよう切り直してください。")


# ─── run ─────────────────────────────────────────────────────────────────────


def _parse_modes(text: str) -> tuple[str, ...] | None:
    """`幻霊,シーズン` のような指定を系統キーの並びにする。

    重複は最初の位置だけ残す。1つも取れなければ None。
    """
    from afkj import states as st

    keys: list[str] = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        spec = st.MODES_BY_CLI_NAME.get(name)
        if spec is None:
            return None
        if spec.key not in keys:
            keys.append(spec.key)
    return tuple(keys) or None


def _parse_attempts(text: str) -> tuple[int, ...] | None:
    """`3,2` のような指定を、各クリア編成での挑戦回数にする。"""
    values: list[int] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        if not item.isdigit() or int(item) < 1:
            return None
        values.append(int(item))
    return tuple(values) or None


def cmd_run(args: argparse.Namespace) -> int:
    from afkj import states as st
    from afkj import window as win
    from afkj.runner import RunConfig, Runner
    from afkj.vision import TemplateStore

    mode_order = _parse_modes(args.modes)
    if mode_order is None:
        names = " / ".join(m.cli_name for m in st.MODES)
        print(f"[エラー] --modes の指定が不正です: {args.modes}")
        print(f"        使えるのは {names} をカンマで並べたもの（例: 幻霊,シーズン）")
        return 1

    attempts = _parse_attempts(args.attempts)
    if attempts is None:
        print(f"[エラー] --attempts の指定が不正です: {args.attempts}")
        print("        1以上の整数をカンマで並べてください（例: 3,2）")
        return 1

    store = TemplateStore(TEMPLATES_DIR)
    store.load()
    if not store.templates:
        print(f"[エラー] テンプレートが登録されていません: {store.index_path}")
        print("        `afkj capture` → `afkj crop` の順に実行してください。")
        return 1

    missing = store.missing_files()
    if missing:
        print(f"[エラー] 画像ファイルが見つかりません: {', '.join(missing)}")
        return 1

    if not win.is_admin() and not args.dry_run:
        print("[警告] 管理者権限がありません。クリックがゲームに届かない可能性があります。")
        print("       ターミナルを右クリック →「管理者として実行」で開き直すことを推奨します。")

    game = win.GameWindow(WINDOW_TITLE)
    if not game.attach():
        print(f"[エラー] ゲームウィンドウ '{WINDOW_TITLE}' が見つかりません。")
        return 1

    config = RunConfig(
        mode_order=mode_order,
        formation_attempts=attempts,
        max_battles=args.max_battles,
        max_runtime=args.max_minutes * 60 if args.max_minutes else 0.0,
        dry_run=args.dry_run,
        record=args.record,
        debug_dir=DEBUG_DIR,
    )
    runner = Runner(game, store, config)

    try:
        reason = runner.loop()
    except KeyboardInterrupt:
        reason = "ユーザーによる中断 (Ctrl+C)"

    print()
    print("=" * 64)
    print(f"  停止: {reason}")
    print(f"  {runner.stats.summary()}")
    for line in runner.stats.mode_report():
        print(f"  {line}")
    print(f"  {runner.stage_tracking_report()}")
    print("=" * 64)
    return 0


# ─── エントリーポイント ──────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afkj",
        description="AFK Journey 自動操作ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細ログを出す")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="環境を診断する")
    p_doctor.set_defaults(func=cmd_doctor)

    p_capture = sub.add_parser("capture", help="ゲーム画面を撮影する（F9で保存 / F10で終了）")
    p_capture.add_argument("--out", help=f"保存先フォルダ (既定: {SCREENSHOTS_DIR.name}/)")
    p_capture.add_argument("--prefix", default="shot", help="ファイル名の接頭辞 (既定: shot)")
    p_capture.add_argument(
        "--manual",
        action="store_true",
        help="自動保存をやめ、F9 を押したときだけ撮る",
    )
    p_capture.add_argument(
        "--interval", type=float, default=0.7, help="画面を見に行く間隔・秒 (既定: 0.7)"
    )
    p_capture.set_defaults(func=cmd_capture)

    p_crop = sub.add_parser("crop", help="スクショからテンプレート画像を切り出す")
    p_crop.add_argument("--src", help=f"読み込むフォルダ (既定: {SCREENSHOTS_DIR.name}/)")
    p_crop.set_defaults(func=cmd_crop)

    p_check = sub.add_parser("check", help="今の画面に各テンプレートがどれだけ一致するか調べる")
    p_check.set_defaults(func=cmd_check)

    p_run = sub.add_parser("run", help="自動操作を開始する")
    p_run.add_argument(
        "--modes",
        default="幻霊,シーズン",
        metavar="幻霊,シーズン",
        help=(
            "回す先鋒ステージの系統とその順番。前のものを限界まで粘ってから"
            "次へ移る (既定: 幻霊,シーズン。片方だけなら 幻霊 のように書く)"
        ),
    )
    p_run.add_argument(
        "--attempts",
        default="3,2",
        metavar="X,Y",
        help=(
            "各クリア編成での挑戦回数。3,2 なら1番目の編成で3回、"
            "それでも勝てなければ2番目の編成で2回試して諦める (既定: 3,2)"
        ),
    )
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="画面の判定だけしてクリックしない（動作確認用）",
    )
    p_run.add_argument(
        "--record",
        action="store_true",
        help="状態が変わるたびに画面を debug_screens/ に保存する（判定の検証用）",
    )
    p_run.add_argument(
        "--max-battles", type=int, default=0, help="この回数だけ戦ったら終了 (既定: 無制限)"
    )
    p_run.add_argument(
        "--max-minutes", type=int, default=0, help="この分数で終了 (既定: 無制限)"
    )
    p_run.set_defaults(func=cmd_run)

    return parser


def main() -> int:
    setup_console_utf8()
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n中断しました。")
        sys.exit(130)
