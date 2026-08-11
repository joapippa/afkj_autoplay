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

ステージを1つでも進めていたら、負けた相手は別の（より強い）ステージなので
1番目のクリア編成から数え直す。進めたかどうかは次の2つを観測できたかで
判断する（推測ではなく観測に基づくよう、根拠をログに残す）。

    - オート戦闘終了（集計）画面 … 1つ以上クリアしたときだけ出る
    - 戦闘勝利の画面

勝利数の計上に使っているステージ看板の変化は、根拠に含めない。
過大に数えることが実測で分かっており（集計画面が4クリアと出た周回で9回）、
誤って「進めた」と判断すると同じ編成で永久に粘ってしまう。

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

from . import states as st
from . import window as win
from .vision import Frame, TemplateStore, imwrite_unicode, to_gray

log = logging.getLogger(__name__)

# これ以上一致していれば、その画面で確定として残りの照合を省く。
# 優先度順に調べているので、より優先度の低い状態が結果を覆すことはない。
CONFIDENT_SCORE = 0.93


def _for_saving(frame: Frame) -> np.ndarray:
    """保存用の画像。カラーがあればカラー（BGR）、なければ白黒。"""
    color = getattr(frame, "color", None)
    if color is not None:
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    return frame.gray


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """2つの切り出し画像の正規化相関（-1〜1）。

    平均を引いてから比べるので、明るさの違いには左右されない。
    背景の演出が多少動いても、同じ看板なら高い値になる。
    """
    if a.shape != b.shape:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom else 0.0


# ─── 設定 ────────────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """実行時の各種設定。"""

    # 回す系統とその順番。前のものを限界までやってから次へ移る。
    mode_order: tuple[str, ...] = st.DEFAULT_MODE_ORDER
    # 各クリア編成での挑戦回数。(3, 2) なら1番目の編成で3回、
    # それでも勝てなければ2番目の編成で2回試し、そこで諦める。
    formation_attempts: tuple[int, ...] = (3, 2)
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


@dataclass
class Stats:
    """実行結果の集計。系統ごとに分けて持つ。"""

    started_at: float = field(default_factory=time.time)
    clicks: int = 0
    recoveries: int = 0
    unknown_frames: int = 0
    # 系統キー → その系統の成績。最初に触れた順で並ぶ。
    modes: dict[str, ModeStats] = field(default_factory=dict)

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
            state = f"諦めた（{s.give_up_reason}）" if s.given_up else "まだ余力あり"
            lines.append(
                f"{label}: 勝ち {s.victories} 負け {s.defeats} / {state}"
            )
        return lines or ["系統ごとの記録なし（戦闘に入りませんでした）"]


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

        # 有効な状態（テンプレートが揃っているものだけ）
        self.states = self._enabled_states()
        self._states_by_priority = sorted(self.states, key=lambda s: -s.priority)

        # 進行状況のメモ。編成画面では「クリア編成を押したか」で
        # 次に押すボタンが変わるため、これを覚えておく必要がある。
        self.formation_applied = False

        # ── 系統（幻霊 / シーズン）の進行管理 ──────────────────────────
        # 今どちらを回しているか。ステージ選択画面でボタンを押し分ける。
        self._mode: str = config.mode_order[0] if config.mode_order else st.PHANTOM
        # 限界まで粘って諦めた系統。以降は回さない。
        self._exhausted: set[str] = set()
        # 諦めた直後で、ステージ選択へ戻る途中かどうか。
        self._giving_up = False
        self._giving_up_since = 0.0

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
        # 今開いているポップアップで「>」を押した回数。
        self._arrow_presses = 0

        # 同じ画面に留まっているかの検出用
        self._last_state_key: str | None = None
        self._state_changed: bool = False
        self._state_since: float = time.time()
        self._escalation: int = 0
        self._unknown_streak: int = 0
        self._last_signature: bytes | None = None
        self._frozen_since: float = time.time()

        # ステージ看板の見た目。変化したら1ステージ進んだと数える。
        self._stage_ref: np.ndarray | None = None
        self._stage_pending: np.ndarray | None = None
        # 看板が写っていたコマ / 写っていなかったコマの数（集計が合わないときの手掛かり）
        self._stage_present = 0
        self._stage_absent = 0

    # ── 準備 ──────────────────────────────────────────────────────────────

    def check_action_templates(self) -> None:
        """操作に使うテンプレートの欠けを、何ができなくなるかとあわせて知らせる。

        状態判定用のテンプレート（states.py の detect）は _enabled_states が
        面倒を見るが、押すボタンのほうは判定に出てこないので気づけない。
        """
        for name, effect in st.OPTIONAL_ACTION_TEMPLATES.items():
            if name not in self.store.templates:
                log.warning("テンプレート『%s』が未登録です → %s", name, effect)

    def _enabled_states(self) -> list[st.StateSpec]:
        enabled: list[st.StateSpec] = []
        for state in st.STATES:
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
            state, match = self._identify(screen_gray)
            self._track(state, screen_gray)
            self._record_transition(state, screen_gray)
            self._track_stage_progress(state, screen_gray)

            if state is None:
                self._handle_unknown(screen_gray, rect)
                continue

            self._unknown_streak = 0
            self._act(state, match, screen_gray, rect)

    def _mode_labels(self) -> list[str]:
        return [
            st.MODES_BY_KEY[k].label for k in self.config.mode_order if k in st.MODES_BY_KEY
        ]

    def _attempts_plan_text(self) -> str:
        return " → ".join(
            f"{i + 1}番目のクリア編成で{n}回"
            for i, n in enumerate(self.config.formation_attempts)
        ) + " → 諦めて次の系統へ"

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
        """
        mode = self._pick_mode()
        if mode is None:
            return  # ループ先頭の停止判定で止まる

        if mode != self._mode:
            log.info(
                "[%s] %s は諦めたので %s に移ります",
                state.label, st.MODES_BY_KEY[self._mode].label, st.MODES_BY_KEY[mode].label,
            )
            self._mode = mode

        self._giving_up = False
        self.formation_applied = False
        self._start_new_stage("ステージ選択に戻った")

        spec = st.MODES_BY_KEY[mode]
        self.stats.mode(mode)  # 一度も勝ち負けしなくても内訳に出るように
        log.info("[%s] %s へ入ります（%s）", state.label, spec.label, spec.entry_template)
        if not self._click_template(spec.entry_template, screen_gray, rect):
            log.warning("  %s が見つかりません（別系統のボタンは押しません）", spec.entry_template)
            self._escalate(rect)

    # ── 系統の切り替え ────────────────────────────────────────────────────

    def _pick_mode(self) -> str | None:
        """まだ諦めていない系統のうち、指定順で先にあるもの。無ければ None。"""
        for key in self.config.mode_order:
            if key not in self._exhausted:
                return key
        return None

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
            log.info("★ すべての系統を回し切りました。ステージ選択へ戻って終了します")
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
        # 系統を移ると看板の文字そのものが変わる（幻霊 / シーズン）。
        # 前の看板を基準に残すと、切り替わりを1ステージ進んだと数えてしまう。
        self._reset_stage_tracking()

    def _note_progress(self, evidence: str) -> None:
        """ステージを進めた証拠を記録する。

        呼ぶのは信頼できる観測だけ（オート戦闘終了の集計画面と勝利画面）。
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
        return 0

    def _on_formation(self, state, match, screen_gray, rect) -> None:
        """編成画面。

        ボタンは左から ← / クリア編成 / オート挑戦 / 戦闘。
        「戦闘」は手動で戦うためのボタンなので絶対に押さない。押すのは
        「オート挑戦」のほう。候補にも入れていない。

        クリア編成 → 一括適用 → オート挑戦 の順に進むが、一括適用のあとは
        再びこの画面に戻ってくる。そのため「すでに編成を適用したか」を
        覚えておき、次に押すボタンを切り替える。
        なお同じ画面で足踏みが続いた場合は、覚えている状態が実態と
        ずれている可能性があるので、もう一方のボタンも試す。
        """
        if self._giving_up:
            # 諦めたので ← でステージ選択へ戻る。
            # ここで画面中央下をタップすると「戦闘」（手動戦闘）を
            # 踏みかねないので、← が見つからないときは何もしない。
            log.info("[%s] 諦めたので ← でステージ選択へ戻ります", state.label)
            if not self._click_template("戻る_btn", screen_gray, rect):
                log.warning("  ← が見つかりません。少し待って押し直します")
                time.sleep(self.config.poll_interval)
            return

        if self._stuck_for() > self.config.stuck_seconds:
            self.formation_applied = not self.formation_applied
            self.stats.recoveries += 1
            log.warning("[%s] 進まないので押すボタンを切り替えます", state.label)
            self._state_since = time.time()

        if self.formation_applied:
            order = ["オート挑戦_btn", "クリア編成_btn"]
            log.info("[%s] オート挑戦を押します", state.label)
        else:
            order = ["クリア編成_btn", "オート挑戦_btn"]
            log.info("[%s] クリア編成を押します", state.label)

        self._click_first_available(order, screen_gray, rect)

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

        if self._giving_up:
            log.info("[%s] 諦めたのでポップアップを閉じます", state.label)
            self._tap_dismiss(rect)
            return

        if self.formation_applied:
            log.info("[%s] 適用済み。ポップアップを閉じます", state.label)
            self._tap_dismiss(rect)
            return

        if self._arrow_presses < self._formation_index:
            log.info(
                "[%s] %d番目のクリア編成へ送ります（> を %d/%d回）",
                state.label, self._formation_index + 1,
                self._arrow_presses + 1, self._formation_index,
            )
            if self._click_template("次の編成_btn", screen_gray, rect):
                self._arrow_presses += 1
            else:
                # 「>」が無い＝これ以上クリア編成が無い。ここが本当の限界。
                log.info("  > が見つかりません。これ以上のクリア編成はありません")
                self._give_up_mode(f"クリア編成が{self._formation_index}個で尽きた")
            return

        log.info(
            "[%s] %d番目のクリア編成を一括適用します（この編成で %d/%d回目）",
            state.label, self._formation_index + 1,
            self._attempts + 1, self._allowed_attempts(),
        )
        if self._click_template("一括適用_btn", screen_gray, rect):
            self.formation_applied = True
        else:
            log.warning("  一括適用が見つかりません")
            self._escalate(rect)

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

        log.info("[%s] 周回中... (%d秒)", state.label, elapsed)
        time.sleep(self.config.battle_interval)

    def _on_loading(self, state, match, screen_gray, rect) -> None:
        log.info("[%s] 読み込みを待ちます (%d秒)", state.label, int(self._stuck_for()))
        time.sleep(self.config.poll_interval)

    def _on_auto_battle_end(self, state, match, screen_gray, rect) -> None:
        # この画面はどこをタップしても閉じる。閉じると敗北画面へ進む。
        #
        # この集計画面は「1つ以上クリアしてから途切れた」ときだけ出る
        # （一度も勝てずに負けたときは出ない）。つまりこの画面が見えた
        # ことが、ステージを進めた証拠になる。
        log.info("[%s] タップして閉じます", state.label)
        self._note_progress("オート戦闘終了（集計）画面を観測")
        self.formation_applied = False
        self._reset_stage_tracking()
        self._tap_dismiss(rect)

    def _on_defeat(self, state, match, screen_gray, rect) -> None:
        if self._state_changed:
            # 敗北画面は数秒表示され続けるので、切り替わった最初の1回だけ
            # 数えて、粘り方も1回だけ進める。
            self._register_defeat()

        self.formation_applied = False
        self._reset_stage_tracking()

        if self._giving_up:
            log.info("[%s] 諦めたので ← でステージ選択へ戻ります", state.label)
            if not self._click_template("戻る_btn", screen_gray, rect):
                log.warning("  ← が見つかりません。少し待って押し直します")
                time.sleep(self.config.poll_interval)
            return

        log.info(
            "[%s] もう一度を押して再挑戦します（%d番目のクリア編成で %d/%d回目）",
            state.label, self._formation_index + 1,
            self._attempts + 1, self._allowed_attempts(),
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

        self.stats.mode(self._mode).defeats += 1
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
        if self._formation_index >= len(plan):
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

        オート周回が続いている間は、ゲームが自動で次のステージへ進むので
        何もしないで待つのが正しい。ここで余計なタップをすると、次の戦闘の
        画面を触ってしまう。
        周回が終わっている場合は結果パネルに『挑戦』ボタンが出るので、
        それを押して次へ進む。
        """
        # 勝った回数はここでは数えない。ステージ看板の変化で数えている
        # （勝利画面は表示が短く、判定の間隔によっては見逃すため）。
        # ただし「進めたかどうか」の判断には使う。数え落としても
        # 集計画面・看板の変化のどれかを観測できれば足りる。
        self._note_progress("戦闘勝利の画面を観測")

        if self._auto_run_active(screen_gray):
            log.info("[%s] オート周回が継続中。自動で次へ進むのを待ちます", state.label)
            time.sleep(self.config.poll_interval)
            return

        log.info("[%s] オート周回は終了。次へ進みます", state.label)
        self.formation_applied = False
        entry = st.MODES_BY_KEY[self._mode].entry_template
        if not self._click_template(entry, screen_gray, rect):
            self._tap_dismiss(rect)

    def _auto_run_active(self, screen_gray: np.ndarray) -> bool:
        """オート周回が続いているか。

        「オート挑戦中」の表示は戦闘中だけでなく勝利画面やロード画面でも
        出ており、周回が途切れると消える。勝利後に待つべきか進めるべきかを
        これで見分ける。
        """
        if "オート挑戦中" not in self.store.templates:
            return False
        return self.store.find("オート挑戦中", screen_gray) is not None

    # ── ステージ進行の追跡（勝利数の計上）────────────────────────────────

    def _track_stage_progress(self, state: st.StateSpec | None, frame: Frame) -> None:
        """ステージの看板が変わったかを見て、勝った回数を数える。

        ★ 未完成。実走では集計画面が「合計2/4ステージクリア」と表示した
          周回で、ここでの計上が 0 だった。原因は切り分け中で、
          戦闘中の判定タイミングで看板が写るコマを拾えていない疑いがある。
          stage_tracking_report() の観測率で切り分けられるようにしてある。
          周回の動作そのものには影響しない（表示される数値だけの問題）。
        """
        if state is None or state.key != st.IN_BATTLE:
            return

        sample = self._stage_label(frame)
        if sample is None:
            self._stage_absent += 1
            log.debug("看板ステータス: 写っていない (通算 %d回)", self._stage_absent)
            return

        self._stage_present += 1

        if self._stage_ref is None:
            self._stage_ref = sample
            log.debug("看板ステータス: 基準を取得 (通算 %d回)", self._stage_present)
            return

        score = _correlation(self._stage_ref, sample)
        log.debug("看板ステータス: 基準との相関 %.3f (しきい値 %.2f)",
                  score, st.STAGE_SAME_CORRELATION)

        if score >= st.STAGE_SAME_CORRELATION:
            return  # 同じステージのまま

        # 看板が変わった＝ステージが進んだ。
        # 看板が写っているコマ同士だけを比べているので、
        # 同じステージなら相関はきわめて高くなる（実測 0.97）。
        # そのため1回の変化で判断してよい。
        # ★ ここでの計上は周回の制御に使わない（表示だけ）。
        #   実走で、集計画面が「合計4ステージクリア（54→58）」と出た周回で
        #   ここは 9回 数えていた。この過大計上を「ステージを進めた」根拠に
        #   混ぜると、同じステージに負け続けているのに挑戦回数が数え直しに
        #   なり、いつまでも1番目のクリア編成で粘って諦めなくなる。
        #   根拠は集計画面と勝利画面だけに絞っている（_note_progress の呼び先）。
        self.stats.mode(self._mode).victories += 1
        self._stage_ref = sample
        log.info("ステージが進みました (相関 %.3f) → 通算: %s", score, self.stats.summary())

    def _stage_label(self, frame: Frame) -> np.ndarray | None:
        """ステージ看板の領域を切り出す。看板が出ていなければ None。

        オート挑戦を押した直後の約2秒と、勝利後に次のステージへ移る間は
        ロード画面や場面転換の演出になっていて看板が出ていない。
        その状態のコマを比較に混ぜると、番号が変わったのか看板が消えたのか
        区別がつかず、数え間違いの原因になる。
        白い文字がどれだけあるかで、看板が出ているコマだけを選ぶ。
        """
        x1r, y1r, x2r, y2r = st.STAGE_LABEL_REGION
        h, w = frame.gray.shape[:2]
        x1, x2 = int(w * x1r), int(w * x2r)
        y1, y2 = int(h * y1r), int(h * y2r)
        if x2 - x1 < 8 or y2 - y1 < 4:
            return None

        crop = frame.gray[y1:y2, x1:x2]
        text_ratio = float((crop > st.STAGE_LABEL_BRIGHTNESS).mean())
        if text_ratio < st.STAGE_LABEL_MIN_TEXT_RATIO:
            return None  # 看板が出ていない（ロード中・場面転換中）
        return crop.astype(np.float32)

    def _reset_stage_tracking(self) -> None:
        """周回が途切れたら基準を捨てる（次の周回を新しく数え始める）。"""
        self._stage_ref = None
        self._stage_pending = None

    def stage_tracking_report(self) -> str:
        """看板をどれだけ観測できたかの内訳。集計が合わないときの手掛かり。"""
        total = self._stage_present + self._stage_absent
        if total == 0:
            return "看板の観測: 戦闘中の判定が一度も行われませんでした"
        rate = self._stage_present / total
        return (
            f"看板の観測: 写っていた {self._stage_present}回 / "
            f"写っていなかった {self._stage_absent}回 (観測率 {rate:.0%})"
        )

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

    def _click_first_available(
        self, names: list[str], screen_gray: np.ndarray, rect: win.WindowRect
    ) -> bool:
        """候補のうち最初に見つかったものをクリックする。

        ★ 系統（幻霊 / シーズン）のエントリーボタンには使わないこと。
          押し分けが必要なので、候補を並べると別系統に入ってしまう。
        """
        for name in names:
            if self._click_template(name, screen_gray, rect):
                return True

        log.warning("  押せるボタンが見つかりません (%s)", ", ".join(names))
        self._escalate(rect)
        return False

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
        """
        self._unknown_streak += 1
        self.stats.unknown_frames += 1

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
