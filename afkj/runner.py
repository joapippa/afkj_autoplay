"""
自動操作の本体（状態機械）
==========================
毎周回で「画面を撮る → 今どの状態か判定する → その状態に応じた操作をする」
を繰り返す。手順を固定で持たないので、次のような場面でも自力で復帰する。

    - クリックが空振りした    → 画面が変わらないので次の周回でまた押す
    - 想定外のポップアップ    → 不明状態として ESC や閉じる操作を試す
    - 画面遷移が遅れた        → 遷移するまで待ってから進む
    - ゲームが再起動された    → ウィンドウを取り直して続行

周回の方針（「限界まで頑張り切る」）
------------------------------------
先鋒ステージは幻霊とシーズンの2系統があり、進捗は独立している。
1つの系統について、こう粘る:

    1番目のクリア編成で X回 挑む
      → それでも負けるなら 2番目のクリア編成で Y回
        → 設定した編成をすべて使い切ったら、その系統は諦めて次の系統へ
          → どちらも諦めたら実行終了

戦闘は「オート挑戦」ではなく「戦闘」で1戦ずつ行う。勝ったら ← で
ステージ選択へ戻り、次のステージでもクリア編成の1番目から当て直す。
オート挑戦だと勝った編成のまま次へ進むが、その勝率は 27/82 = 33% で、
ステージごとのクリア編成1番目（50/64 = 78%）に大きく劣った。

ステージを1つでも進めていたら、負けた相手は別の（より強い）ステージなので
1番目のクリア編成から数え直す。進めたかどうかは戦闘勝利の画面を観測できたかで
判断する（推測ではなく観測に基づくよう、根拠をログに残す）。

クリアしたステージは勝利画面ごとに1つ数え、画面に出ている
「先鋒ステージ進捗 161 ≫ 162」を読んで全体の進み具合を残す（digits.py）。

停止方法:
    Ctrl+C / F10 キー / マウスを画面左上角へ移動
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import digits
from . import ocr
from . import states as st
from . import window as win
from .vision import Frame, TemplateStore, imwrite_unicode, to_gray

log = logging.getLogger(__name__)

# これ以上一致していれば、その画面で確定として残りの照合を省く。
# 優先度順に調べているので、より優先度の低い状態が結果を覆すことはない。
CONFIDENT_SCORE = 0.93

# ステージ選択で「今の系統のボタンが無く、別系統のボタンだけが見える」が
# この回数続いたら、その系統はクリアし終えてボタンが消えたと判断する。
# 1回で決めないのは、演出や一瞬の見落としで系統を捨てないため。
ENTRY_ABSENT_FRAMES = 3

# クリア編成のポップアップで「>」が見つからないことがこの回数続いたら、
# クリア編成が尽きたと判断する。1回で決めると、送った直後の演出などで
# 一瞬見落としただけで実行全体が終わってしまう（結社戦線は1フロアに
# 数十戦かけるので、取り違えたときの損が大きい）。
ARROW_ABSENT_FRAMES = 3

# 結社戦線の画面でフロア番号を読む回数の上限。画面が滑り込んでくる途中の
# フレームでは番号の帯がまだ所定の位置に無く、読めない（実測: 挑戦ボタンが
# 41px 下にずれていたフレームで失敗し、2秒前の同じ画面では読めていた）。
FLOOR_READ_TRIES = 3

# 「戦闘」を押してからこの秒数までは、判定できない画面も戦闘中とみなして
# タップせずに待つ。戦闘画面のテンプレートは撮影済みのオート挑戦の画面から
# 切り出したもので、1戦ずつの戦闘でも効くかはまだ実走で確かめていないため。
# 戦闘の制限時間は90秒（画面右上のカウント）なので、読み込みを含めて余裕を見る。
BATTLE_GRACE_SECONDS = 150.0

# 戦闘中に判定できなかった画面を保存する上限（テンプレートを切り出す材料）。
BATTLE_UNKNOWN_SAVES = 3

# Skip は1戦につき1回だけ押す。この秒数たっても戦闘画面に Skip が見えていたら
# （押したクリックが届かなかった）押し直す。
# 早く押し直すのは危ない。Skip を押してから勝敗の画面が出るまで10秒ほどかかり、
# 押し直したクリックが敗北画面の同じ位置（進捗の帯）に落ちると、もう一度ではなく
# 結社戦線の画面へ戻ってしまう（2026-10-17 07:34〜07:36 の実走で3回起きた）。
SKIP_RETRY_SECONDS = 30.0

# 「戦闘」を押してから、まだ編成画面が見えていても押し直さずに待つ秒数。
# 押してから戦闘画面になるまで毎回 6〜9秒かかり、押した回数（2〜4回）に
# よらなかった（2026-10-17 07:48〜07:50 の9戦）。最初の1回で届いている。
BATTLE_CLICK_WAIT = 12.0

# Skip を押すまでの判定間隔（秒）。早く押すほど1戦が短くなる。
# Skip だけを探すなら1回 0.03〜0.1秒で済む（画面全体の判定は1〜2.8秒）。
SKIP_POLL_SECONDS = 0.3

# 編成ポップアップを閉じるときにタップする位置（画面サイズに対する割合）。
# ポップアップの外側で、編成画面に戻っていた場合でも何もない場所（右上の木々）。
# 画面下の中央（_tap_dismiss の位置）は編成画面の『オート挑戦』の上なので使わない
# （2026-10-17 07:33 にポップアップが閉じた後にタップが届き、オート挑戦が始まった）。
POPUP_DISMISS_POINT = (1700 / 1920, 450 / 1200)


def _for_saving(frame: Frame) -> np.ndarray:
    """保存用の画像。カラーがあればカラー（BGR）、なければ白黒。"""
    color = getattr(frame, "color", None)
    if color is not None:
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    return frame.gray


# ─── 設定 ────────────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """実行時の各種設定。"""

    # 回す系統とその順番。前のものを限界までやってから次へ移る。
    mode_order: tuple[str, ...] = st.DEFAULT_MODE_ORDER
    # 各クリア編成での挑戦回数。(3, 2, 1, 1, 1) なら1番目の編成で3回、
    # 2番目で2回、3〜5番目で1回ずつ試し、そこで諦める。
    # 深く粘るより広く浅く（README「値の決め方」参照）。
    formation_attempts: tuple[int, ...] = (3, 2, 1, 1, 1)
    # True なら最後の値を、クリア編成が尽きるまで繰り返す。(20, 10) なら
    # 1番目で20回、2番目以降は「>」が無くなるまで10回ずつ（結社戦線の既定）。
    repeat_last_attempts: bool = False
    poll_interval: float = 1.2  # 通常時の判定間隔（秒）
    battle_interval: float = 3.0  # 戦闘中の判定間隔（秒）
    after_click: float = 1.6  # クリック後に画面が変わるのを待つ時間（秒）
    max_battles: int = 0  # 0 なら無制限
    max_runtime: float = 0.0  # 0 なら無制限（秒）
    stuck_seconds: float = 25.0  # 同じ画面が続いたら手を打つまでの時間
    # 諦めてからステージ選択に戻れないまま経った時間の上限（秒）。
    # ← が押せない画面に閉じ込められて、黙って回り続けるのを防ぐ。
    give_up_timeout: float = 90.0
    unknown_limit: int = 40  # 不明状態が連続してよい回数
    dry_run: bool = False  # 判定だけしてクリックしない
    record: bool = False  # 状態が変わるたびに画面を保存する（検証用）
    debug_dir: Path = Path("debug_screens")


@dataclass
class ModeStats:
    """1つの系統（幻霊 / シーズン）ごとの集計。"""

    victories: int = 0
    defeats: int = 0
    given_up: bool = False
    give_up_reason: str = ""
    # ステージ選択にこの系統のボタンが無かった（クリアし終えて消えた）。
    absent: bool = False
    # クリア編成ごとの勝ち負け。キーは 0 始まりの番号（1番目の編成なら 0）。
    formation_wins: dict[int, int] = field(default_factory=dict)
    formation_defeats: dict[int, int] = field(default_factory=dict)

    def formation_report(self) -> str:
        """「1番目 勝ち 2 負け 5」のように編成ごとに並べる。"""
        used = sorted(set(self.formation_wins) | set(self.formation_defeats))
        return " / ".join(
            f"{i + 1}番目 勝ち {self.formation_wins.get(i, 0)} 負け {self.formation_defeats.get(i, 0)}"
            for i in used
        )


@dataclass
class Stats:
    """実行結果の集計。系統ごとに分けて持つ。"""

    started_at: float = field(default_factory=time.time)
    clicks: int = 0
    recoveries: int = 0
    unknown_frames: int = 0
    # 系統キー → その系統の成績。最初に触れた順で並ぶ。
    modes: dict[str, ModeStats] = field(default_factory=dict)
    # 勝利画面で読んだ「先鋒ステージ進捗」。最初と最後、および読めた回数。
    progress_first: int | None = None
    progress_last: int | None = None
    progress_read: int = 0
    progress_unread: int = 0
    # 回しているコンテンツ（先鋒ステージ / 結社戦線）。進捗の出し方が違う。
    content: str = st.SENPO
    # 結社戦線のフロア番号（結社戦線の画面の「現在：N」を読んだもの）。
    floor_first: int | None = None
    floor_last: int | None = None
    floor_unread: int = 0
    # クリアしたフロア: (フロア番号 or None, 勝った編成の番号 1〜, そのフロアで何戦目か)
    floors_cleared: list[tuple[int | None, int, int]] = field(default_factory=list)

    def note_progress_numbers(self, before: int, after: int) -> None:
        if self.progress_first is None:
            self.progress_first = before
        self.progress_last = after
        self.progress_read += 1

    def mode(self, key: str) -> ModeStats:
        return self.modes.setdefault(key, ModeStats())

    @property
    def victories(self) -> int:
        return sum(m.victories for m in self.modes.values())

    @property
    def defeats(self) -> int:
        return sum(m.defeats for m in self.modes.values())

    @property
    def battles(self) -> int:
        return self.victories + self.defeats

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def summary(self) -> str:
        mins, secs = divmod(int(self.elapsed()), 60)
        hours, mins = divmod(mins, 60)
        return (
            f"経過 {hours}:{mins:02d}:{secs:02d} / "
            f"戦闘 {self.battles}回 (勝ち {self.victories} 負け {self.defeats}) / "
            f"クリック {self.clicks}回 / 自動復帰 {self.recoveries}回"
        )

    def mode_report(self) -> list[str]:
        """系統ごとの内訳。どちらをどこまで粘ったかを残す。"""
        lines = []
        for key, s in self.modes.items():
            spec = st.MODES_BY_KEY.get(key)
            label = spec.label if spec else key
            if s.absent:
                state = "ステージ選択にボタンが無いため回さず（クリア済みとみなした）"
            elif s.given_up:
                state = f"諦めた（{s.give_up_reason}）"
            else:
                state = "まだ余力あり"
            lines.append(
                f"{label}: 勝ち {s.victories} 負け {s.defeats} / {state}"
            )
            if s.formation_wins or s.formation_defeats:
                lines.append(f"  クリア編成ごと: {s.formation_report()}")
        return lines or ["系統ごとの記録なし（戦闘に入りませんでした）"]

    def progress_report(self) -> str:
        """勝利画面から読んだ進捗。ゲーム自身が出している数字。"""
        if self.content == st.UNION:
            return self.floor_report()
        total = self.progress_read + self.progress_unread
        if not total:
            return "先鋒ステージ進捗: 勝利画面が出ませんでした（1ステージもクリアせず）"
        read = f"勝利画面 {total}回中 {self.progress_read}回 読み取り"
        if self.progress_first is None:
            return f"先鋒ステージ進捗: 読み取れませんでした（{read}）"
        return (
            f"先鋒ステージ進捗: {self.progress_first} → {self.progress_last} "
            f"(+{self.progress_last - self.progress_first}) / {read}"
        )

    def floor_report(self) -> str:
        """結社戦線のフロアの進み具合と、クリアしたフロアの内訳。"""
        if self.floor_first is None:
            head = "結社戦線のフロア: 読み取れませんでした"
        else:
            head = f"結社戦線のフロア: {self.floor_first} → {self.floor_last}"
        if self.floor_unread:
            head += f"（読み取れなかった回数 {self.floor_unread}）"
        if not self.floors_cleared:
            return head + " / クリアしたフロアなし"
        detail = ", ".join(
            f"{'?' if floor is None else floor}（{index}番目の編成・{battles}戦目）"
            for floor, index, battles in self.floors_cleared
        )
        return f"{head} / クリア {len(self.floors_cleared)}: {detail}"


class StopReason:
    USER = "ユーザーによる停止"
    FAILSAFE = "フェイルセーフ（マウスが左上角）"
    MAX_BATTLES = "指定した戦闘回数に到達"
    MAX_RUNTIME = "指定した実行時間に到達"
    LOST_WINDOW = "ゲームウィンドウを見失った"
    TOO_MANY_UNKNOWN = "画面を判定できない状態が続いた"
    ALL_MODES_DONE = "すべての系統を限界まで回し切った"
    GIVE_UP_STUCK = "諦めたあとステージ選択へ戻れなかった"


# ─── 実行本体 ────────────────────────────────────────────────────────────────


class Runner:
    def __init__(self, game: win.GameWindow, store: TemplateStore, config: RunConfig):
        self.game = game
        self.store = store
        self.config = config
        self.stats = Stats()

        # 回すコンテンツ。系統の並びはすべて同じコンテンツのもの（afkj.py で確かめている）。
        first = config.mode_order[0] if config.mode_order else st.PHANTOM
        self.content = st.MODES_BY_KEY[first].content
        self.stats.content = self.content

        # 有効な状態（テンプレートが揃っているものだけ）
        self.states = self._enabled_states()
        self._states_by_priority = sorted(self.states, key=lambda s: -s.priority)

        # 進行状況のメモ。編成画面では「クリア編成を押したか」で
        # 次に押すボタンが変わるため、これを覚えておく必要がある。
        self.formation_applied = False
        # 最後に一括適用したクリア編成の番号（0 始まり）。「もう一度」で編成画面へ
        # 戻ると当てた編成が残っているので、同じ番号で挑み直すなら当て直さない。
        # ステージが変わったときと、編成画面の外から入り直したときは None に戻す。
        self._applied_index: int | None = None

        # ── 系統（幻霊 / シーズン）の進行管理 ──────────────────────────
        # 今どちらを回しているか。ステージ選択画面でボタンを押し分ける。
        self._mode: str = config.mode_order[0] if config.mode_order else st.PHANTOM
        # 限界まで粘って諦めた系統。以降は回さない。
        self._exhausted: set[str] = set()
        # 諦めた直後で、ステージ選択へ戻る途中かどうか。
        self._giving_up = False
        self._giving_up_since = 0.0
        # ステージ選択で「今の系統のボタンは無いが、別系統のボタンは見える」
        # が続いたフレーム数。系統をクリアし終えるとそのボタンは消える。
        self._entry_absent_frames = 0

        # ── クリア編成の使い分け ────────────────────────────────────────
        # index 番目のクリア編成で formation_attempts[index] 回まで挑む。
        self._formation_index = 0
        self._attempts = 0
        # このステージで1つでも先へ進めたか（進めていれば編成を数え直す）。
        self._progressed = False
        self._progress_evidence = ""
        # 前回数えた敗北のあと、戦闘を観測したか。
        # 敗北画面 → 一瞬の判定不能 → 敗北画面 という揺れがあり
        # （記録にも auto_battle_end → unknown → defeat の並びがある）、
        # これを2回の敗北として数えると挑戦回数を余計に消費してしまう。
        self._battle_observed = True
        self._last_defeat_at = 0.0
        # 「戦闘」を押した時刻（戦闘が終わったら 0）。戦闘中の判定できない画面を
        # タップしないために使う。
        self._battle_started_at = 0.0
        self._battle_unknown_saved = 0
        # 最後に「戦闘」を押した時刻。読み込み中に押し直さないために使う。
        self._battle_clicked_at = 0.0
        # この戦闘で Skip を押したか、と最後に押した時刻
        self._skip_pressed = False
        self._skip_clicked_at = 0.0
        # 今開いているポップアップで「>」を押した回数。
        self._arrow_presses = 0
        # 「>」が続けて見つからなかった回数。
        self._arrow_missing = 0
        # このステージ（フロア）で戦った回数。勝ったときに「何戦目か」を残す。
        self._stage_battles = 0
        # 結社戦線の画面で読んだフロア番号（読めなければ None）。
        self._floor: int | None = None
        # 結社戦線の画面に入ってから番号を読みにいった回数と、入る前の番号。
        self._floor_tries = 0
        self._floor_before: int | None = None
        # 最後に読めたフロア番号のあとに勝った回数。次に読む番号の妥当な範囲を決める。
        self._wins_since_floor = 0

        # 同じ画面に留まっているかの検出用
        self._last_state_key: str | None = None
        self._state_changed: bool = False
        self._state_since: float = time.time()
        self._escalation: int = 0
        self._unknown_streak: int = 0
        self._last_signature: bytes | None = None
        self._frozen_since: float = time.time()

        # 集計画面の「先鋒ステージ進捗」を読む係
        self.digits = digits.DigitReader(store.directory / "digits")

    # ── 準備 ──────────────────────────────────────────────────────────────

    def check_action_templates(self) -> None:
        """操作に使うテンプレートの欠けを、何ができなくなるかとあわせて知らせる。

        状態判定用のテンプレート（states.py の detect）は _enabled_states が
        面倒を見るが、押すボタンのほうは判定に出てこないので気づけない。
        """
        actions = dict(st.OPTIONAL_ACTION_TEMPLATES)
        if self.content == st.UNION:
            actions.update(st.UNION_ACTION_TEMPLATES)
        for name, effect in actions.items():
            if name not in self.store.templates:
                log.warning("テンプレート『%s』が未登録です → %s", name, effect)

    def _enabled_states(self) -> list[st.StateSpec]:
        enabled: list[st.StateSpec] = []
        for state in st.states_for(self.content):
            available = [n for n in state.detect if n in self.store.templates]
            if not available:
                log.warning(
                    "状態『%s』はテンプレート未登録のため無効です (%s)",
                    state.label, ", ".join(state.detect),
                )
                continue
            if len(available) < len(state.detect):
                missing = set(state.detect) - set(available)
                log.info("状態『%s』: %s が未登録（残りで判定します）", state.label, ", ".join(missing))
            enabled.append(state)
        return enabled

    # ── メインループ ──────────────────────────────────────────────────────

    def loop(self) -> str:
        """停止するまで回し続ける。停止理由を返す。"""
        log.info("=" * 60)
        log.info("自動操作 開始%s", " / 判定のみ・クリックしません" if self.config.dry_run else "")
        log.info("コンテンツ: %s", st.CONTENT_LABELS[self.content])
        log.info("回す順番: %s", " → ".join(self._mode_labels()))
        log.info("各ステージの粘り方: %s", self._attempts_plan_text())
        log.info("停止: Ctrl+C / F10 / マウスを画面左上角へ")
        log.info("=" * 60)
        self.check_action_templates()

        while True:
            reason = self._check_stop_conditions()
            if reason:
                return reason

            frame = self._capture()
            if frame is None:
                # ウィンドウを見失っても即あきらめず、復帰を待つ
                if not self._wait_for_window():
                    return StopReason.LOST_WINDOW
                continue

            screen_gray, rect = frame
            # 戦闘画面の Skip は画面全体の判定（1〜2.8秒）を待たずに押す。
            # 判定に時間がかかると、押す時点の画面が撮った画面と違ってしまう。
            if self._try_skip(screen_gray, rect):
                continue

            state, match = self._identify(screen_gray)
            self._track(state, screen_gray)
            self._record_transition(state, screen_gray)

            if state is None:
                self._handle_unknown(screen_gray, rect)
                continue

            self._unknown_streak = 0
            self._act(state, match, screen_gray, rect)

    def _entry_screen_label(self) -> str:
        """← で戻る先の画面（ログ用）。"""
        return "結社戦線の画面" if self.content == st.UNION else "ステージ選択"

    def _mode_labels(self) -> list[str]:
        return [
            st.MODES_BY_KEY[k].label for k in self.config.mode_order if k in st.MODES_BY_KEY
        ]

    def _attempts_plan_text(self) -> str:
        plan = self.config.formation_attempts
        text = " → ".join(f"{i + 1}番目のクリア編成で{n}回" for i, n in enumerate(plan))
        if self.config.repeat_last_attempts:
            text += f" → 以降の編成も各{plan[-1]}回（クリア編成が尽きるまで）"
        return text + " → 諦めて次の系統へ"

    def _check_stop_conditions(self) -> str | None:
        if win.is_key_down(win.VK_F10):
            return StopReason.USER
        if win.failsafe_triggered():
            return StopReason.FAILSAFE
        if self.config.max_battles and self.stats.battles >= self.config.max_battles:
            return StopReason.MAX_BATTLES
        if self.config.max_runtime and self.stats.elapsed() >= self.config.max_runtime:
            return StopReason.MAX_RUNTIME
        if self._unknown_streak >= self.config.unknown_limit:
            return StopReason.TOO_MANY_UNKNOWN
        if self._pick_mode() is None:
            return StopReason.ALL_MODES_DONE
        if (
            self._giving_up
            and self.config.give_up_timeout
            and time.time() - self._giving_up_since > self.config.give_up_timeout
        ):
            # ← が押せない画面に閉じ込められている。黙って回り続けない。
            log.warning(
                "諦めてから %.0f秒 たってもステージ選択へ戻れませんでした",
                self.config.give_up_timeout,
            )
            return StopReason.GIVE_UP_STUCK
        return None

    def _capture(self) -> tuple[Frame, win.WindowRect] | None:
        """画面を撮って照合用の Frame にする。縮小は1周期に1回だけ行う。"""
        shot = self.game.capture()
        if shot is None:
            return None
        rgb, rect = shot
        frame = Frame(to_gray(rgb))
        # 記録用に元のカラー画像も持たせておく（白黒だと画面を見分けにくい）
        frame.color = rgb
        return frame, rect

    def _wait_for_window(self, timeout: float = 60.0) -> bool:
        """ゲームウィンドウが戻ってくるのを待つ（再起動などに備える）。"""
        log.warning("ゲームウィンドウを見失いました。%.0f秒まで復帰を待ちます...", timeout)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if win.is_key_down(win.VK_F10) or win.failsafe_triggered():
                return False
            time.sleep(2.0)
            self.game.hwnd = None  # 掴み直す
            if self.game.attach():
                log.info("ゲームウィンドウが戻りました。続行します。")
                self.stats.recoveries += 1
                return True
        return False

    # ── 状態判定 ──────────────────────────────────────────────────────────

    def _identify(self, screen_gray: np.ndarray):
        """今の画面がどの状態かを判定する。

        しきい値を超えた状態のうち、優先度→スコアの順で最良のものを選ぶ。
        「最初に見つかったもの」ではなく「最も一致したもの」を採るのが要点。
        複数の状態が重なって見える場面（編成画面の上に一覧が出るなど）で
        取り違えないため。
        """
        best_state = None
        best_match = None
        best_rank = None
        hits: dict[str, float] = {}

        # 優先度の高いものから調べる。高優先度の状態が強く一致したら、
        # それより低い優先度の状態は結果を覆せないので調べずに切り上げる。
        for state in self._states_by_priority:
            match = self.store.find_best(
                [n for n in state.detect if n in self.store.templates], screen_gray
            )
            if match is None or match.score < state.threshold:
                continue
            hits[state.key] = match.score
            rank = (state.priority, match.score)
            if best_rank is None or rank > best_rank:
                best_rank, best_state, best_match = rank, state, match
            if match.score >= CONFIDENT_SCORE:
                break

        self._warn_if_contradictory(hits)
        return best_state, best_match

    def _warn_if_contradictory(self, hits: dict[str, float]) -> None:
        """同時に成立しないはずの状態が両方見えていたら知らせる。

        勝利と敗北が同時に出ることはないので、両方が反応したときは
        テンプレートがその2画面を見分けられていないということ。
        黙って高い方を採ると、勝ったのに敗北として数え続けることになる。
        """
        for group in st.EXCLUSIVE_GROUPS:
            matched = {k: hits[k] for k in group if k in hits}
            if len(matched) < 2:
                continue
            detail = " / ".join(
                f"{st.STATES_BY_KEY[k].label}={v:.3f}" for k, v in matched.items()
            )
            log.warning(
                "同時に成立しないはずの画面が両方反応しています (%s)。"
                "テンプレートが見分けられていない可能性があります", detail
            )
            log.warning(
                "  → `afkj check` を勝敗それぞれの画面で実行し、"
                "片方だけが反応するよう切り直してください"
            )

    def _track(self, state: st.StateSpec | None, screen_gray: np.ndarray) -> None:
        """同じ画面に留まっていないか、画面が固まっていないかを記録する。"""
        key = state.key if state else "__unknown__"
        self._state_changed = key != self._last_state_key
        if self._state_changed:
            self._last_state_key = key
            self._state_since = time.time()
            self._escalation = 0
            self._entry_absent_frames = 0

        # 画面が動いているか（戦闘中はアニメで常に変わる）
        signature = cv2.resize(screen_gray.work, (16, 16), interpolation=cv2.INTER_AREA).tobytes()
        if signature != self._last_signature:
            self._last_signature = signature
            self._frozen_since = time.time()

    def _stuck_for(self) -> float:
        return time.time() - self._state_since

    # ── 状態ごとの操作 ────────────────────────────────────────────────────

    def _act(self, state: st.StateSpec, match, screen_gray: np.ndarray, rect: win.WindowRect) -> None:
        handler = {
            st.STAGE_SELECT: self._on_stage_select,
            st.FORMATION: self._on_formation,
            st.CLEAR_FORMATION_LIST: self._on_clear_formation_list,
            st.IN_BATTLE: self._on_in_battle,
            st.LOADING: self._on_loading,
            st.AUTO_BATTLE_END: self._on_auto_battle_end,
            st.DEFEAT: self._on_defeat,
            st.VICTORY: self._on_victory,
            st.UNION_TOP: self._on_union_top,
            st.GUILD_MODES: self._on_guild_modes,
            st.MODE_MENU: self._on_mode_menu,
            st.WORLD_MAP: self._on_world_map,
            st.WORLD: self._on_world,
        }.get(state.key)

        if handler is None:
            log.warning("状態『%s』の操作が未定義です", state.label)
            time.sleep(self.config.poll_interval)
            return

        handler(state, match, screen_gray, rect)

    def _on_stage_select(self, state, match, screen_gray, rect) -> None:
        """ステージ選択画面。今回そうする系統のボタンだけを押す。

        ★ ここでボタンをフォールバックさせてはいけない。「幻霊挑戦」が
          見つからないときに「挑戦」を押すと、別系統（シーズン先鋒）を
          回してしまい、粘り方の管理も勝敗の内訳もずれる。
          見つからないなら押さずに、次の周回で押し直す。

        ただし系統をクリアし終えるとそのボタンは画面から消え、残った系統の
        「挑戦」だけが中央に1つ出る。「別系統のボタンは見えるのに、今の系統の
        ボタンが無い」が ENTRY_ABSENT_FRAMES 回続いたら、その系統は無いものと
        して次へ移る（押し直しても出てこないので、待つと永久に詰まる）。
        """
        mode = self._pick_mode()
        if mode is None:
            return  # ループ先頭の停止判定で止まる

        if mode != self._mode:
            log.info(
                "[%s] %s は終えたので %s に移ります",
                state.label, st.MODES_BY_KEY[self._mode].label, st.MODES_BY_KEY[mode].label,
            )
            self._mode = mode

        self._giving_up = False
        self.formation_applied = False
        self._battle_started_at = 0.0
        self._start_new_stage("ステージ選択に戻った")

        spec = st.MODES_BY_KEY[mode]
        self.stats.mode(mode)  # 一度も勝ち負けしなくても内訳に出るように
        entry = (
            self.store.find(spec.entry_template, screen_gray)
            if spec.entry_template in self.store.templates else None
        )
        if entry is not None:
            log.info("[%s] %s へ入ります（%s）", state.label, spec.label, spec.entry_template)
            self._entry_absent_frames = 0
            self._click_at(entry.center, rect, label=spec.entry_template, score=entry.score)
            return

        others = self._visible_other_entries(mode, screen_gray)
        if not others:
            log.warning("  %s が見つかりません（別系統のボタンは押しません）", spec.entry_template)
            self._escalate(rect)
            return

        # 別系統のボタンだけが見えている。この系統はクリアし終えて消えた可能性が高い。
        # 一瞬の見落としで系統を捨てないよう、続けて確認できてから判断する。
        self._entry_absent_frames += 1
        seen = ", ".join(f"{m.name}={m.score:.3f}" for m in others)
        log.info(
            "[%s] %s が無く、%s だけが見えます（確認 %d/%d）",
            state.label, spec.entry_template, seen,
            self._entry_absent_frames, ENTRY_ABSENT_FRAMES,
        )
        if self._entry_absent_frames >= ENTRY_ABSENT_FRAMES:
            self._skip_absent_mode(mode)
        else:
            time.sleep(self.config.poll_interval)

    # ── 系統の切り替え ────────────────────────────────────────────────────

    def _pick_mode(self) -> str | None:
        """まだ諦めていない系統のうち、指定順で先にあるもの。無ければ None。"""
        for key in self.config.mode_order:
            if key not in self._exhausted:
                return key
        return None

    def _visible_other_entries(self, mode: str, screen_gray: np.ndarray) -> list:
        """ステージ選択画面に見えている、今の系統以外のエントリーボタン。"""
        found = []
        for spec in st.MODES:
            if spec.content != self.content:
                continue  # 結社戦線の「挑戦」は先鋒の「挑戦」と見分けられない
            if spec.key == mode or spec.entry_template not in self.store.templates:
                continue
            match = self.store.find(spec.entry_template, screen_gray)
            if match is not None:
                found.append(match)
        return found

    def _skip_absent_mode(self, mode: str) -> None:
        """ステージ選択にボタンが無い系統を飛ばす。

        諦める（_give_up_mode）のと違い、すでにステージ選択にいるので
        ← で戻る必要はない。次のフレームで残りの系統のボタンを押す。
        """
        self._entry_absent_frames = 0
        self._exhausted.add(mode)
        self.stats.mode(mode).absent = True
        log.info(
            "★ %s のボタンがありません（クリア済みとみなして飛ばします）",
            st.MODES_BY_KEY[mode].label,
        )
        remaining = self._pick_mode()
        if remaining is not None:
            log.info("★ 次は %s を回します", st.MODES_BY_KEY[remaining].label)

    def _give_up_mode(self, reason: str) -> None:
        """今の系統を諦める。ステージ選択へ戻って次の系統へ移る。"""
        mode = self._mode
        if mode in self._exhausted:
            return
        self._exhausted.add(mode)
        stats = self.stats.mode(mode)
        stats.given_up = True
        stats.give_up_reason = reason
        label = st.MODES_BY_KEY[mode].label
        log.info("★ %s はここまで（%s）", label, reason)

        if "戻る_btn" not in self.store.templates:
            # ← が押せないとステージ選択へ戻れない。当てずっぽうにタップすると
            # 編成画面では「戦闘」を押してしまう位置なので、触らずに終了する。
            log.warning(
                "戻る_btn が未登録のためステージ選択へ戻れません。実行を終了します"
            )
            self._exhausted.update(self.config.mode_order)
            return

        self._giving_up = True
        self._giving_up_since = time.time()
        remaining = self._pick_mode()
        if remaining is None:
            log.info("★ すべての系統を回し切りました。ここで終了します")
        else:
            log.info("★ 次は %s を回します", st.MODES_BY_KEY[remaining].label)

    # ── クリア編成の使い分け ──────────────────────────────────────────────

    def _start_new_stage(self, reason: str) -> None:
        """このステージの粘り方を最初から数え直す。"""
        if self._formation_index or self._attempts:
            log.debug("粘り方をリセット（%s）", reason)
        self._formation_index = 0
        self._attempts = 0
        self._progressed = False
        self._progress_evidence = ""
        self._arrow_presses = 0
        self._stage_battles = 0
        # 別のステージにはそのステージのクリア編成を当て直す
        self._applied_index = None

    def _note_progress(self, evidence: str) -> None:
        """ステージを進めた証拠を記録する。

        呼ぶのは信頼できる観測だけ（勝利画面と、オート戦闘終了の集計画面）。
        看板の変化による勝利数の計上は過大なので、ここには入れない。
        """
        if not self._progressed:
            log.debug("ステージを進めた証拠: %s", evidence)
        self._progressed = True
        self._progress_evidence = evidence

    def _allowed_attempts(self) -> int:
        plan = self.config.formation_attempts
        if self._formation_index < len(plan):
            return max(1, plan[self._formation_index])
        if self.config.repeat_last_attempts and plan:
            return max(1, plan[-1])
        return 0

    def _on_formation(self, state, match, screen_gray, rect) -> None:
        """編成画面。

        ボタンは左から ← / クリア編成 / オート挑戦 / 戦闘。
        押すのは「戦闘」（1戦だけ戦う）。「オート挑戦」は勝つと編成を
        引き継いだまま次のステージへ進んでしまうので、候補にも入れていない。

        クリア編成 → 一括適用 → 戦闘 の順に進むが、一括適用のあとは
        再びこの画面に戻ってくる。そのため「すでに編成を適用したか」を
        覚えておき、次に押すボタンを切り替える。
        なお同じ画面で足踏みが続いた場合は、覚えている状態が実態と
        ずれている可能性があるので、もう一方のボタンも試す。
        """
        self._battle_started_at = 0.0  # 戦闘を始められていない
        if self._giving_up:
            # 諦めたので ← でステージ選択へ戻る。
            # ここで画面中央下をタップすると「戦闘」を
            # 踏みかねないので、← が見つからないときは何もしない。
            log.info("[%s] 諦めたので ← で%sへ戻ります", state.label, self._entry_screen_label())
            if not self._click_template("戻る_btn", screen_gray, rect):
                log.warning("  ← が見つかりません。少し待って押し直します")
                time.sleep(self.config.poll_interval)
            return

        if self._stuck_for() > self.config.stuck_seconds:
            self.formation_applied = not self.formation_applied
            self.stats.recoveries += 1
            log.warning("[%s] 進まないので押すボタンを切り替えます", state.label)
            self._state_since = time.time()

        # 押すボタンは1つだけ。見つからなければ押さずに待つ。
        # ・代わりに別のボタンを押さない。以前は「戦闘」が見つからないと「クリア編成」を
        #   押していて、画面が滑り込む途中のフレームで編成ポップアップを開き直していた。
        # ・_escalate も呼ばない。画面下の中央をタップするので、ここでは『オート挑戦』を押す。
        # 覚えている状態が実態とずれていたら、上の足踏みの切り替えで直る。
        if self.formation_applied:
            since = time.time() - self._battle_clicked_at
            if since < BATTLE_CLICK_WAIT:
                log.info("[%s] 戦闘を押しました。読み込みを待ちます (%.0f秒)", state.label, since)
                self._battle_started_at = self._battle_clicked_at
                time.sleep(self.config.poll_interval)
                return
            name = "戦闘_btn"
            log.info("[%s] 戦闘を押します", state.label)
        else:
            # クリア編成が見つからなくても戦闘は押さない（編成を当てずに戦うことになる）。
            name = "クリア編成_btn"
            log.info("[%s] クリア編成を押します", state.label)

        if not self._click_template(name, screen_gray, rect):
            log.info("  %s がまだ見えません。少し待ちます", name)
            time.sleep(self.config.poll_interval)
            return
        if name == "戦闘_btn":
            # 戦闘画面を判定できなくても、この戦闘の勝敗は1回として数える
            self._battle_observed = True
            self._battle_started_at = self._battle_clicked_at = time.time()
            self._skip_pressed = False

    def _on_clear_formation_list(self, state, match, screen_gray, rect) -> None:
        """クリア編成のポップアップ。

        一括適用を押したあともポップアップが残ることがあるので、
        一度押したあとは画面をタップして閉じにいく。押し続けて
        同じ場所で足踏みするのを防ぐ。

        2番目以降のクリア編成を使うときは、右端の「>」を必要な回数だけ
        押してから一括適用する。ポップアップを開き直すと1番目に戻る前提で、
        開いてからの押した回数を数える（状態が変わった＝開き直したとみなす）。
        1回の周回で1つずつ押すので、押した結果が画面に出てから次へ進む。
        """
        if self._state_changed:
            self._arrow_presses = 0
            self._arrow_missing = 0

        if self._giving_up:
            log.info("[%s] 諦めたのでポップアップを閉じます", state.label)
            self._dismiss_popup(state, rect)
            return

        if self.formation_applied:
            self._dismiss_popup(state, rect)
            return

        if self._arrow_presses < self._formation_index:
            log.info(
                "[%s] %d番目のクリア編成へ送ります（> を %d/%d回）",
                state.label, self._formation_index + 1,
                self._arrow_presses + 1, self._formation_index,
            )
            if self._click_template("次の編成_btn", screen_gray, rect):
                self._arrow_presses += 1
                self._arrow_missing = 0
                return
            # 「>」が無い＝これ以上クリア編成が無い。ここが本当の限界。
            # ただし一瞬の見落としで諦めないよう、続けて確認できてから決める。
            self._arrow_missing += 1
            if self._arrow_missing < ARROW_ABSENT_FRAMES:
                log.info(
                    "  > が見つかりません（確認 %d/%d）",
                    self._arrow_missing, ARROW_ABSENT_FRAMES,
                )
                time.sleep(self.config.poll_interval)
                return
            log.info("  > が見つかりません。これ以上のクリア編成はありません")
            self._give_up_mode(f"クリア編成が{self._formation_index}個で尽きた")
            return

        log.info(
            "[%s] %d番目のクリア編成を一括適用します（この編成で %d/%d回目）",
            state.label, self._formation_index + 1,
            self._attempts + 1, self._allowed_attempts(),
        )
        if self.config.record and self._formation_index:
            # 「>」で送ったあとの画面は状態が変わらないので _record_transition では
            # 残らない。本当に別の編成を当てたかを後で確かめられるよう、ここで残す。
            path = self.config.debug_dir / (
                f"rec_{time.strftime('%H%M%S')}_apply_{self._formation_index + 1}.png"
            )
            if imwrite_unicode(path, _for_saving(screen_gray)):
                log.info("  記録: %s", path.name)
        if self._click_template("一括適用_btn", screen_gray, rect):
            self.formation_applied = True
            self._applied_index = self._formation_index
        else:
            log.warning("  一括適用が見つかりません")
            self._escalate(rect)

    def _dismiss_popup(self, state, rect: win.WindowRect) -> None:
        """一括適用のあとに残った編成ポップアップを閉じる。

        一括適用を押すとポップアップは閉じていく途中なので、判定に使った画面は
        もう古いことがある。タップする直前に撮り直し、まだ開いているときだけ、
        編成画面に戻っていても何も起きない位置（POPUP_DISMISS_POINT）をタップする。
        """
        fresh = self._capture()
        if fresh is None:
            return
        frame, rect = fresh
        if self.store.find("一括適用_btn", frame) is None:
            log.info("[%s] ポップアップはもう閉じています", state.label)
            return
        log.info("[%s] 適用済み。ポップアップの外をタップして閉じます", state.label)
        if self.config.dry_run:
            log.info("  [判定のみ] タップしません")
            time.sleep(self.config.poll_interval)
            return
        fx, fy = POPUP_DISMISS_POINT
        x, y = int(rect.width * fx), int(rect.height * fy)
        sx, sy = rect.to_screen(x, y)
        log.info("  画面をタップ: 画面内(%d,%d) → 絶対(%d,%d)", x, y, sx, sy)
        self.game.focus()
        win.click(sx, sy)
        self.stats.clicks += 1
        time.sleep(self.config.after_click)

    def _on_in_battle(self, state, match, screen_gray, rect) -> None:
        # 「前回の敗北のあとに戦った」ことの目印。敗北を二重に数えないために使う。
        self._battle_observed = True
        elapsed = int(self._stuck_for())
        frozen = time.time() - self._frozen_since

        # 周回中は画面が動き続けるはず。固まっていたら結果画面を見落として
        # いる可能性があるので、いったん画面をタップして先へ促す。
        if frozen > 25.0:
            log.warning("[%s] 画面が %.0f秒 動いていません。タップして進めます", state.label, frozen)
            self._tap_dismiss(rect)
            self.stats.recoveries += 1
            self._frozen_since = time.time()
            return

        if self._skip_pressed:
            log.info("[%s] Skip を押しました。結果の画面を待ちます (%d秒)", state.label, elapsed)
        else:
            log.info("[%s] 戦っています... (%d秒)", state.label, elapsed)
        time.sleep(self._battle_wait())

    def _battle_wait(self) -> float:
        """戦闘中に次の判定までに待つ秒数。Skip を押すまでは短く刻む。"""
        if not self._skip_pressed and "スキップ_btn" in self.store.templates:
            return SKIP_POLL_SECONDS
        return self.config.poll_interval if self._skip_pressed else self.config.battle_interval

    def _try_skip(self, screen_gray: Frame, rect: win.WindowRect) -> bool:
        """戦闘画面の Skip が見えたら押す。押したら True。

        Skip を押すとすぐに勝敗の画面へ進む（演出を含めて10秒ほど）。押すのは
        1戦につき1回だけ。押し直しは SKIP_RETRY_SECONDS たっても Skip が
        見えているときに限る（理由は定数の説明を参照）。
        Skip は戦闘画面にしか出ないので、画面全体の判定より先に調べてよい
        （Skip の無い画面での一致度は最大 0.644、しきい値は 0.85）。
        """
        if "スキップ_btn" not in self.store.templates:
            return False
        since = time.time() - self._skip_clicked_at
        if self._skip_pressed and since < SKIP_RETRY_SECONDS:
            return False
        skip = self.store.find("スキップ_btn", screen_gray)
        if skip is None:
            return False
        if self._skip_pressed:
            log.warning("Skip を押して %.0f秒 たっても戦闘画面のままです。押し直します", since)
        else:
            started = time.time() - self._battle_started_at if self._battle_started_at else 0.0
            log.info("[戦闘中] Skip で戦闘を飛ばします（戦闘を押して %.0f秒）", started)
        self._click_at(skip.center, rect, label="スキップ_btn", score=skip.score)
        self._skip_pressed = True
        self._skip_clicked_at = time.time()
        # 「前回の敗北のあとに戦った」ことの目印（_on_in_battle と同じ）
        self._battle_observed = True
        return True

    def _on_loading(self, state, match, screen_gray, rect) -> None:
        log.info("[%s] 読み込みを待ちます (%d秒)", state.label, int(self._stuck_for()))
        time.sleep(self.config.poll_interval)

    def _on_auto_battle_end(self, state, match, screen_gray, rect) -> None:
        # オート挑戦の終了集計。オート挑戦は押さないので本来は出ないが、
        # 出たらタップして閉じる（どこをタップしても閉じる）。
        # この画面は1つ以上クリアしたときだけ出るので、進めた証拠にはなる。
        # クリア数は勝利画面で数えているので、ここでは足さない。
        log.info("[%s] タップして閉じます", state.label)
        self._note_progress("オート戦闘終了（集計）画面を観測")
        self.formation_applied = False
        self._tap_dismiss(rect)

    def _read_progress(self, screen_gray: Frame) -> None:
        """勝利画面の「先鋒ステージ進捗 A ≫ B」を読み、進捗の記録に残す。"""
        if not self.digits.available:
            return
        got = self.digits.read_progress(screen_gray.gray, digits.VICTORY_PROGRESS_REGION)
        if got is None:
            self.stats.progress_unread += 1
            log.info("  進捗の数字を読み取れませんでした")
            return

        before, after = got
        if after - before != 1:
            # 1戦で進むのは1ステージだけ。違えば読み違い（演出が数字に重なるなど）
            self.stats.progress_unread += 1
            log.warning("  進捗の数字が不自然です (%d → %d)。読み飛ばします", before, after)
            return

        self.stats.note_progress_numbers(before, after)
        log.info("  先鋒ステージ進捗 %d → %d", before, after)

    def _on_defeat(self, state, match, screen_gray, rect) -> None:
        if self._state_changed:
            # 敗北画面は数秒表示され続けるので、切り替わった最初の1回だけ
            # 数えて、粘り方も1回だけ進める。
            self._register_defeat()

        # 「もう一度」で戻った編成画面には当てた編成が残っている。
        # 同じ編成で挑み直すなら、クリア編成を開かずにそのまま「戦闘」を押す。
        same = self._applied_index is not None and self._applied_index == self._formation_index
        self.formation_applied = same
        self._battle_started_at = self._battle_clicked_at = 0.0
        self._skip_pressed = False

        if self._giving_up:
            log.info("[%s] 諦めたので ← で%sへ戻ります", state.label, self._entry_screen_label())
            if not self._click_template("戻る_btn", screen_gray, rect):
                log.warning("  ← が見つかりません。少し待って押し直します")
                time.sleep(self.config.poll_interval)
            return

        log.info(
            "[%s] もう一度を押して再挑戦します（%d番目のクリア編成で %d/%d回目・%s）",
            state.label, self._formation_index + 1,
            self._attempts + 1, self._allowed_attempts(),
            "編成はそのまま" if same else "編成を当て直す",
        )
        if not self._click_template("もう一度_btn", screen_gray, rect):
            log.warning("  もう一度が見つかりません")
            self._escalate(rect)

    def _register_defeat(self) -> None:
        """負けた1回を数え、次にどのクリア編成で挑むかを決める。

        ステージを進めていたなら、負けた相手は前と別のステージなので
        1番目のクリア編成から数え直す。進めていなければ同じステージに
        負け続けているということなので、回数を使い切ったら次の編成へ移り、
        編成も使い切ったらこの系統は諦める。
        """
        # 同じ敗北を二重に数えない。ただし「戦闘を観測できなかった」だけを
        # 根拠に見送ると、戦闘の判定をたまたま撮り逃したときに回数が
        # 増えなくなり、同じ編成で永久に粘ってしまう。時間でも見る。
        since = time.time() - self._last_defeat_at
        if not self._battle_observed and since < 25.0:
            log.info(
                "  この敗北は %.0f秒前と同じものとみて数えません"
                "（戦闘を観測していないため）", since,
            )
            return
        self._battle_observed = False
        self._last_defeat_at = time.time()

        mode_stats = self.stats.mode(self._mode)
        mode_stats.defeats += 1
        index = self._formation_index
        mode_stats.formation_defeats[index] = mode_stats.formation_defeats.get(index, 0) + 1
        self._stage_battles += 1
        log.info("  → 通算: %s", self.stats.summary())

        if self._progressed:
            log.info(
                "  ステージを進めていたので1番目のクリア編成から数え直します（根拠: %s）",
                self._progress_evidence,
            )
            self._start_new_stage("ステージが進んでいた")
            return

        self._attempts += 1
        allowed = self._allowed_attempts()
        if self._attempts < allowed:
            log.info(
                "  同じステージに負けました。%d番目のクリア編成で %d/%d回目を試します",
                self._formation_index + 1, self._attempts + 1, allowed,
            )
            return

        # この編成では回数を使い切った。次の編成へ。
        self._formation_index += 1
        self._attempts = 0
        self._arrow_presses = 0
        plan = self.config.formation_attempts
        if self._formation_index >= len(plan) and not self.config.repeat_last_attempts:
            # 繰り返す指定なら、ここでは諦めない。クリア編成の「>」が無くなった
            # ところ（_on_clear_formation_list）が限界になる。
            self._give_up_mode(
                "クリア編成 " + "＋".join(f"{n}回" for n in plan) + " を試し切った"
            )
            return
        log.info(
            "  %d番目のクリア編成では勝てませんでした。%d番目の編成で %d回 試します",
            self._formation_index, self._formation_index + 1, self._allowed_attempts(),
        )

    def _on_victory(self, state, match, screen_gray, rect) -> None:
        """勝利画面。

        結果パネルの下に ← / 統計 / 挑戦 が並ぶ。← でステージ選択へ戻り、
        次のステージでクリア編成を当て直す。『挑戦』は押さない（次の
        ステージへ直接進むはずだが、編成画面を通るかを確かめていない）。

        1戦ずつ戦う場合、勝利画面は ← を押すまで残るので見逃さない。
        そのため勝った回数はここで数える。
        """
        self._battle_started_at = self._battle_clicked_at = 0.0
        self._skip_pressed = False
        if self._state_changed:
            self._register_victory(screen_gray)

        if self._auto_run_active(screen_gray):
            # オート挑戦は押さないので本来は起きない。押されていたら
            # ゲームが自動で次へ進むので、触らずに待つ。
            log.info("[%s] オート周回が継続中。自動で次へ進むのを待ちます", state.label)
            time.sleep(self.config.poll_interval)
            return

        self.formation_applied = False
        log.info(
            "[%s] ← で%sへ戻り、次のステージのクリア編成を当て直します",
            state.label, self._entry_screen_label(),
        )
        if self._click_template("戻る_btn", screen_gray, rect):
            return
        if self._stuck_for() < 10.0:
            # パネルのボタンが出そろう前かもしれない
            log.info("  ← がまだ見えません。少し待ちます")
            time.sleep(self.config.poll_interval)
            return
        # 以前の実行で、勝利画面をタップするとステージ選択へ戻ったのを観測している
        log.warning("  ← が見つかりません。画面をタップして閉じます")
        self._tap_dismiss(rect)

    def _register_victory(self, screen_gray: Frame) -> None:
        """勝った1回を数える。勝利画面が揺れても二重に数えない。

        勝てば次は別のステージなので、ここで粘り方を最初から数え直す。
        先鋒では ← で戻ったステージ選択でも数え直すが、結社戦線の画面では
        数え直さない（_on_union_top）ので、勝った時点で済ませておく。
        """
        if not self._battle_observed:
            log.info("  この勝利は直前と同じものとみて数えません（戦闘を観測していないため）")
            return
        self._battle_observed = False
        mode_stats = self.stats.mode(self._mode)
        mode_stats.victories += 1
        index = self._formation_index
        mode_stats.formation_wins[index] = mode_stats.formation_wins.get(index, 0) + 1
        self._stage_battles += 1

        if self.content == st.UNION:
            floor = self._floor
            self.stats.floors_cleared.append((floor, index + 1, self._stage_battles))
            log.info(
                "  ★ フロア%s クリア（%d番目のクリア編成・このフロア%d戦目）",
                "?" if floor is None else floor, index + 1, self._stage_battles,
            )
            # 次のフロアの番号は結社戦線の画面で読み直す。推測で埋めない。
            self._floor = None
            self._wins_since_floor += 1
        else:
            self._read_progress(screen_gray)
        log.info("  → 通算: %s", self.stats.summary())
        self._start_new_stage("勝った")

    def _auto_run_active(self, screen_gray: np.ndarray) -> bool:
        """オート周回が続いているか。

        「オート挑戦中」の表示は戦闘中だけでなく勝利画面やロード画面でも
        出ており、周回が途切れると消える。勝利後に待つべきか進めるべきかを
        これで見分ける。
        """
        if "オート挑戦中" not in self.store.templates:
            return False
        return self.store.find("オート挑戦中", screen_gray) is not None

    # ── 結社戦線 ──────────────────────────────────────────────────────────

    def _on_union_top(self, state, match, screen_gray, rect) -> None:
        """結社戦線の画面。フロア番号を読んでから『挑戦』を押す。

        先鋒のステージ選択と違い、ここに戻っただけでは粘り方を数え直さない。
        勝ったときに数え直し済みなので、勝たずにここへ戻った（想定外）なら
        同じフロアの続きのはずで、数え直すと同じ編成で永久に粘りかねない。
        ただしフロア番号が変わっていたら、勝利画面を見逃したとみて数え直す。

        『挑戦』が見つからなくても画面をタップしない。_tap_dismiss の位置の
        すぐ下に『掃討』ボタンがあるため。
        """
        mode = self._pick_mode()
        if mode is None:
            return  # ループ先頭の停止判定で止まる
        self._mode = mode
        self._giving_up = False
        self.formation_applied = False
        # 『挑戦』から入り直した編成画面に前の編成が残っているかは確かめていない
        self._applied_index = None
        self._battle_started_at = 0.0
        self.stats.mode(mode)  # 一度も勝ち負けしなくても内訳に出るように

        if self._state_changed:
            self._floor_before = self._floor
            self._floor = None
            self._floor_tries = 0
        if self._floor is None and self._floor_tries < FLOOR_READ_TRIES:
            # 読めなければ挑戦を押さずに待ち、画面が落ち着いてから読み直す
            self._floor_tries += 1
            if not self._read_floor(screen_gray):
                if self._floor_tries < FLOOR_READ_TRIES:
                    log.info(
                        "  フロア番号を読めません。画面が落ち着くのを待って読み直します (%d/%d)",
                        self._floor_tries, FLOOR_READ_TRIES,
                    )
                    time.sleep(self.config.poll_interval)
                    return
                self.stats.floor_unread += 1
                log.info("  フロア番号を読み取れませんでした（%d回試した）", FLOOR_READ_TRIES)
                self._save_floor_screen(screen_gray, "floor_unread")
            before = self._floor_before
            if before is not None and self._floor is not None and self._floor != before:
                log.info("  フロア番号が %d → %d に変わっていました", before, self._floor)
                self._start_new_stage("フロア番号が変わった")

        spec = st.MODES_BY_KEY[mode]
        log.info(
            "[%s] フロア%s に挑戦します",
            state.label, "?" if self._floor is None else self._floor,
        )
        if self._click_template(spec.entry_template, screen_gray, rect):
            return
        log.warning("  %s が見つかりません。少し待って押し直します", spec.entry_template)
        time.sleep(self.config.poll_interval)

    def _read_floor(self, screen_gray: Frame) -> bool:
        """結社戦線の画面の「現在：N」を読む。読めたら True。"""
        value = ocr.read_union_floor(_for_saving(screen_gray))
        if value is None:
            return False

        last = self.stats.floor_last
        if last is not None:
            # 1勝で進むのは1フロアだけで、戻ることもない。範囲を外れた番号は読み違いと
            # みて採らない（実測: 26 をクリアした直後の 27 を「2」と読んだ）。
            # 勝利画面を見逃した場合に備えて、1つ先までは受け入れる。
            high = last + self._wins_since_floor + 1
            if not (last <= value <= high):
                log.warning(
                    "  フロア番号を %d と読みましたが、前回 %d のあと %d勝なので "
                    "%d〜%d のはずです。読み違いとみて採りません",
                    value, last, self._wins_since_floor, last, high,
                )
                self._save_floor_screen(screen_gray, f"floor_misread_{value}")
                return False
        self._wins_since_floor = 0
        if self.stats.floor_first is None:
            self.stats.floor_first = value
        self.stats.floor_last = value
        self._floor = value
        log.info("  フロア番号: %d", value)
        return True

    def _save_floor_screen(self, screen_gray: Frame, label: str) -> None:
        """フロア番号を読めなかった画面を残す（OCR の読み違いを調べる材料）。"""
        path = self.config.debug_dir / f"{time.strftime('%H%M%S')}_{label}.png"
        if imwrite_unicode(path, _for_saving(screen_gray)):
            log.info("  その画面を保存: %s", path.name)

    def _on_guild_modes(self, state, match, screen_gray, rect) -> None:
        log.info("[%s] 結社戦線を開きます", state.label)
        if not self._click_template("結社戦線_card", screen_gray, rect):
            self._escalate(rect)

    def _on_mode_menu(self, state, match, screen_gray, rect) -> None:
        log.info("[%s] ギルドのカテゴリへ移ります", state.label)
        if not self._click_template("ギルド_tab", screen_gray, rect):
            log.warning("  ギルドのカテゴリが見つかりません")
            self._escalate(rect)

    def _on_world_map(self, state, match, screen_gray, rect) -> None:
        log.info("[%s] ← でワールド画面へ戻ります", state.label)
        if not self._click_template("戻る_btn", screen_gray, rect):
            log.warning("  ← が見つかりません")
            self._escalate(rect)

    def _on_world(self, state, match, screen_gray, rect) -> None:
        log.info("[%s] モードを開きます", state.label)
        if not self._click_template("モード_btn", screen_gray, rect):
            self._escalate(rect)

    # ── ステージ進行の追跡（勝利数の計上）────────────────────────────────

    # ── 操作の実行 ────────────────────────────────────────────────────────

    def _click_template(
        self, name: str, screen_gray: np.ndarray, rect: win.WindowRect
    ) -> bool:
        """1つのテンプレートを探してクリックする。見つからなければ False。

        見つからなかったときにどうするかは呼び出し側で決める。
        「押せなかったら待つ」で済む場面と、「押せない＝限界」と判断すべき
        場面（クリア編成の『>』が無い）とがあるため。
        """
        if name not in self.store.templates:
            return False
        match = self.store.find(name, screen_gray)
        if match is None:
            return False
        self._click_at(match.center, rect, label=name, score=match.score)
        return True

    def _click_at(self, center: tuple[int, int], rect: win.WindowRect, label: str, score: float) -> None:
        x, y = center
        if self.config.dry_run:
            log.info("  [判定のみ] %s を検出 (score=%.3f) → クリックはしません", label, score)
            time.sleep(self.config.poll_interval)
            return

        sx, sy = rect.to_screen(x, y)
        log.info("  クリック: %s (score=%.3f) 画面内(%d,%d) → 絶対(%d,%d)", label, score, x, y, sx, sy)
        self.game.focus()
        win.click(sx, sy)
        self.stats.clicks += 1
        time.sleep(self.config.after_click)

    def _tap_dismiss(self, rect: win.WindowRect) -> None:
        """「タップで閉じる」系の画面を、安全な位置をタップして閉じる。

        ボタンではなく余白を狙うため、画面下部の中央やや下を押す。
        """
        if self.config.dry_run:
            log.info("  [判定のみ] 画面をタップして閉じるところ（実行しません）")
            time.sleep(self.config.poll_interval)
            return
        x = rect.width // 2
        y = int(rect.height * 0.90)
        sx, sy = rect.to_screen(x, y)
        log.info("  画面をタップ: 画面内(%d,%d) → 絶対(%d,%d)", x, y, sx, sy)
        self.game.focus()
        win.click(sx, sy)
        self.stats.clicks += 1
        time.sleep(self.config.after_click)

    # ── 詰まったときの立て直し ────────────────────────────────────────────

    def _escalate(self, rect: win.WindowRect) -> None:
        """段階的に手を変えて詰まりを抜ける。

        いきなり強い操作をせず、待つ → 閉じる → ESC の順で試す。
        """
        self._escalation += 1
        self.stats.recoveries += 1
        level = self._escalation

        # ESC は送らない。このゲームで ESC が何を開くか（メニュー、終了確認など）
        # を確認できておらず、周回を壊す恐れがあるため。
        # 画面を閉じる操作は「タップ」で足りることが確認できている。
        if level <= 2:
            log.info("  復帰(%d): 少し待って様子を見ます", level)
            time.sleep(self.config.poll_interval * 2)
        elif level <= 5:
            log.info("  復帰(%d): 画面をタップして閉じてみます", level)
            self._tap_dismiss(rect)
        else:
            log.warning("  復帰(%d): 立て直せません。待機して再試行します", level)
            time.sleep(min(self.config.poll_interval * level, 15.0))

    def _handle_unknown(self, screen_gray: np.ndarray, rect: win.WindowRect) -> None:
        """どの状態にも当てはまらない画面。

        報酬・お知らせ・レベルアップなど想定外のポップアップが典型。
        いきなり乱暴に押さず、段階的に閉じにいく。

        ただし「戦闘」を押した直後は、戦闘画面を判定できていないだけの
        可能性が高い。戦闘中にタップすると英雄のカードを触ってしまうので、
        BATTLE_GRACE_SECONDS までは何もせずに待つ。
        """
        self.stats.unknown_frames += 1
        in_battle = time.time() - self._battle_started_at
        if self._battle_started_at and in_battle < BATTLE_GRACE_SECONDS:
            log.info("画面を判定できません。戦闘中とみて待ちます (%.0f秒)", in_battle)
            if in_battle > 10.0 and self._battle_unknown_saved < BATTLE_UNKNOWN_SAVES:
                self._battle_unknown_saved += 1
                self._save_debug(screen_gray, f"battle_unknown_{self._battle_unknown_saved}")
            time.sleep(self._battle_wait())
            return

        self._unknown_streak += 1

        if self._unknown_streak == 1:
            log.info("画面を判定できません。少し待ちます...")
            time.sleep(self.config.poll_interval * 2)
            return

        if self._unknown_streak in (5, 15, 30):
            self._save_debug(screen_gray, f"unknown_{self._unknown_streak}")

        log.info("画面を判定できません (%d回目)", self._unknown_streak)
        self._escalate(rect)

    def _save_debug(self, screen_gray: Frame, label: str) -> None:
        """判定できなかった画面を保存する（テンプレート追加の材料になる）。"""
        path = self.config.debug_dir / f"{time.strftime('%H%M%S')}_{label}.png"
        if imwrite_unicode(path, _for_saving(screen_gray)):
            log.info("  判定できなかった画面を保存: %s", path)
            log.info("  → このファイルからテンプレートを切り出すと認識できるようになります")

    def _record_transition(self, state: st.StateSpec | None, frame: Frame) -> None:
        """状態が変わるたびに画面を保存する（--record 指定時のみ）。

        ログの状態名だけでは「本当にその画面だったのか」を後から確かめられない。
        遷移の実物を残しておけば、判定が正しかったか検証できる。
        """
        if not self.config.record or not self._state_changed:
            return
        label = state.key if state else "unknown"
        path = self.config.debug_dir / f"rec_{time.strftime('%H%M%S')}_{label}.png"
        if imwrite_unicode(path, _for_saving(frame)):
            log.info("  記録: %s", path.name)
