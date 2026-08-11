"""
画面状態の定義
==============
「この画面ではこのテンプレートが見える」という対応表。

旧版は Step1→2→3→4 の固定手順で、1つでも見つからないと停止していた。
ここでは画面ごとに独立した状態として定義し、実行時は「今どの画面か」を
毎回判定して動く。おかげでクリックが空振りしても、次の周回で同じ画面が
見えるので自然に押し直しになる。

★ ゲームの更新で UI が変わったら、まずここを見直すこと。
   テンプレート名は templates.json のキーと一致させる。


実際の周回（撮影したスクショから確認した流れ）
----------------------------------------------
    ステージ選択 ─[幻霊挑戦/挑戦]→ 編成画面
    編成画面 ─[クリア編成]→ 編成ポップアップ ─[一括適用]→[タップで閉じる]→ 編成画面
    編成画面 ─[オート挑戦]→ 戦闘
    戦闘 ─勝ち→ 戦闘勝利 → そのまま次のステージへ（オート周回が継続）
    戦闘 ─負け→ 戦闘敗北 ─[もう一度]→ 編成画面 へ戻る
    オート周回が途切れる → オート戦闘終了（集計）─[タップ]→ 戦闘敗北
    諦めるとき: 戦闘敗北／編成画面 ─[← 戻る]→ ステージ選択

編成ポップアップには右端に「>」ボタンがあり、押すごとに2番目・3番目…の
クリア編成が表示される。表示中の編成が一括適用の対象になる。

編成画面のボタン列は左から ← / クリア編成 / オート挑戦 / 戦闘。
「戦闘」は手動で戦うためのボタンなので押さないこと。押すのは「オート挑戦」。

「オート挑戦中」の表示は戦闘中だけでなく、勝利画面やロード画面でも
オート周回が続いている限り出ている。逆に周回が終わると消える。
そのため単なる「戦闘中」ではなく「周回が継続中か」の判定に使える。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StateSpec:
    """1つの画面状態。"""

    key: str
    label: str
    # 画面を見分けるための材料。このどれか1つでも見つかればこの画面とみなす。
    #
    # ★ ここには「その画面にしか出ないもの」だけを挙げること。
    #   押したいボタンを一緒に並べてはいけない。たとえば「もう一度」ボタンは
    #   敗北画面にも勝利画面にも出るため、これで敗北を見分けようとすると
    #   勝ったのに敗北と誤判定してしまう。
    #   押すボタンの指定は runner.py 側で別に持つ。
    #
    # ★ 見出しの文字を使うときは、他の画面と違う部分だけを切ること。
    #   「戦闘勝利」と「戦闘敗北」は先頭の「戦闘」が共通なので、
    #   そこを含めると見分けられない。
    detect: tuple[str, ...]
    # 判定に使うしきい値の下限。実際にはテンプレート個別の設定と
    # 高いほうが使われる。
    threshold: float = 0.78
    # 複数の状態が同時に見えるときに、どれを優先するか。
    # 例: 勝利画面には「オート挑戦中」も出ているので、勝利を優先する。
    priority: int = 0
    # 操作せず画面が変わるのを待つ状態
    passive: bool = False
    note: str = ""


@dataclass(frozen=True)
class ModeSpec:
    """先鋒ステージの系統（幻霊 / シーズン）。"""

    key: str
    cli_name: str  # コマンドラインでの指定名
    label: str  # 戦闘画面の看板に出る名前
    entry_template: str  # ステージ選択画面で押すボタン
    note: str = ""


PHANTOM = "phantom"
SEASON = "season"

# 先鋒ステージは「幻霊先鋒ステージ M」と「シーズン先鋒ステージ N」の2つから成り、
# それぞれ独立に進む。ステージ選択画面のボタンで分岐する。
# 画面に出ている先鋒ステージの全体進捗は (M-1) + (N-1)。
#
# ★ ボタンの押し分けを間違えると別系統を回してしまう。実測（1920x1200 の
#   ステージ選択画面）では 幻霊挑戦_btn が x=699 / 挑戦_btn が x=1266 に
#   それぞれ 0.99 / 0.94 で当たり、取り違えは起きていない。
MODES: tuple[ModeSpec, ...] = (
    ModeSpec(
        key=PHANTOM,
        cli_name="幻霊",
        label="幻霊先鋒ステージ",
        entry_template="幻霊挑戦_btn",
        note="ステージ選択画面 左のボタン『幻霊挑戦』から入る",
    ),
    ModeSpec(
        key=SEASON,
        cli_name="シーズン",
        label="シーズン先鋒ステージ",
        entry_template="挑戦_btn",
        note="ステージ選択画面 右の緑ボタン『挑戦』から入る",
    ),
)

MODES_BY_KEY = {m.key: m for m in MODES}
MODES_BY_CLI_NAME = {m.cli_name: m for m in MODES}

# 既定は「幻霊を限界まで → シーズンを限界まで」の順。
DEFAULT_MODE_ORDER: tuple[str, ...] = (PHANTOM, SEASON)


STAGE_SELECT = "stage_select"
FORMATION = "formation"
CLEAR_FORMATION_LIST = "clear_formation_list"
IN_BATTLE = "in_battle"
LOADING = "loading"
AUTO_BATTLE_END = "auto_battle_end"
DEFEAT = "defeat"
VICTORY = "victory"

STATES: tuple[StateSpec, ...] = (
    # ── 結果画面（最優先）───────────────────────────────────────────
    # 勝敗の画面には「オート挑戦中」も一緒に出ているため、
    # 戦闘中と取り違えないよう優先度を高くする。
    StateSpec(
        key=DEFEAT,
        label="戦闘敗北",
        detect=("戦闘敗北",),
        priority=30,
        note="敗北画面。見出しの『敗北』だけで見分ける",
    ),
    StateSpec(
        key=VICTORY,
        label="戦闘勝利",
        detect=("戦闘勝利",),
        priority=30,
        note="勝利画面。見出しの『勝利』だけで見分ける",
    ),
    StateSpec(
        key=AUTO_BATTLE_END,
        label="オート戦闘終了",
        detect=("オート戦闘終了",),
        priority=30,
        note="オート周回の終了集計。タップで閉じると敗北画面へ進む",
    ),
    # ── 操作する画面 ────────────────────────────────────────────────
    StateSpec(
        key=CLEAR_FORMATION_LIST,
        label="編成ポップアップ",
        detect=("一括適用_btn",),
        priority=20,
        note="おすすめ編成のポップアップ。編成画面の上に重なって出る",
    ),
    StateSpec(
        key=FORMATION,
        label="編成画面",
        detect=("クリア編成_btn", "オート挑戦_btn", "戦闘_btn"),
        priority=10,
        note="下部に ← / クリア編成 / オート挑戦 / 戦闘 が並ぶ画面",
    ),
    StateSpec(
        key=STAGE_SELECT,
        label="ステージ選択",
        detect=("幻霊挑戦_btn", "挑戦_btn"),
        priority=5,
        note="幻霊挑戦 / 挑戦 ボタンが見える画面",
    ),
    # ── 待つだけの画面 ──────────────────────────────────────────────
    StateSpec(
        key=IN_BATTLE,
        label="オート周回中",
        detect=("オート挑戦中",),
        passive=True,
        priority=1,
        note="戦闘中・勝利直後・ロード中でも、周回が続いていれば出ている",
    ),
    StateSpec(
        key=LOADING,
        label="ロード中",
        detect=("ロード中",),
        passive=True,
        priority=1,
        note="ロード画面。待てば次の画面になる",
    ),
)


STATES_BY_KEY = {s.key: s for s in STATES}

# 同時には成立しない状態の組。
# 1回の戦闘の結果は勝ちか負けのどちらかで、両方が同時に出ることはない。
# 両方が同時にしきい値を超えたら、それはテンプレートが見分けられていない
# 証拠なので警告を出す（誤ったまま黙って進むより気づけたほうがよい）。
EXCLUSIVE_GROUPS: tuple[tuple[str, ...], ...] = (
    (VICTORY, DEFEAT),
)


# ★ クリアしたステージ数の数え方について
#   以前は戦闘画面の上部にある「幻霊先鋒ステージ N」の看板が変わった回数で
#   数えていたが、**今の版の戦闘画面には看板が出ていない**。あの位置に
#   写っていたのは背景の街並みで、カメラが動くだけで数字が増えていた
#   （実測: 0クリアの周回で3回計上、88クリアの周回で154回計上）。
#   今はオート戦闘終了（集計）画面に出ている数字をそのまま読む → digits.py


def required_templates() -> set[str]:
    """状態判定に使う全テンプレート名。"""
    return {name for state in STATES for name in state.detect}


# 判定には使わないが、操作のために必要なテンプレート。
# 欠けていても周回そのものは回るが、その機能だけが使えなくなる。
# runner.py の起動時に照合し、欠けていれば何ができなくなるかを警告する。
OPTIONAL_ACTION_TEMPLATES: dict[str, str] = {
    "次の編成_btn": "別のクリア編成に切り替えられません（1番目だけで再挑戦します）",
    "戻る_btn": "諦めたときにステージ選択へ戻れません（画面タップで代替します）",
}
