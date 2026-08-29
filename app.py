import html
import io
import os
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

CHART_HEIGHT_PER_ROW = 28
CHART_MIN_HEIGHT = 420

DATE_HEADER_RE = re.compile(r"-+\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일")
MESSAGE_RE = re.compile(
    r"^\[(.+?)\]\s*\[(오전|오후)\s*(\d{1,2}):(\d{2})\]\s?(.*)$"
)
REQUIRED_COLUMNS = ["Date", "User", "Message"]


def decode_upload(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def to_24h(ampm: str, hour: int, minute: int) -> tuple[int, int]:
    if ampm == "오전":
        hour = 0 if hour == 12 else hour
    elif hour != 12:
        hour += 12
    return hour, minute


def is_kakaotalk_text(text: str) -> bool:
    head = text[:3000]
    return "카카오톡 대화" in head or bool(DATE_HEADER_RE.search(head))


def parse_kakaotalk(text: str) -> pd.DataFrame:
    """카카오톡 내보내기(txt)를 Date, User, Message 테이블로 변환합니다."""
    current_date = None
    records: list[dict] = []
    pending: dict | None = None

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        pending["Message"] = pending["Message"].rstrip("\n")
        records.append(pending)
        pending = None

    for line in text.splitlines():
        header = DATE_HEADER_RE.search(line)
        if header:
            flush()
            year, month, day = map(int, header.groups())
            current_date = datetime(year, month, day)
            continue

        match = MESSAGE_RE.match(line)
        if match and current_date is not None:
            flush()
            user, ampm, hour, minute, body = match.groups()
            hour, minute = to_24h(ampm, int(hour), int(minute))
            pending = {
                "Date": current_date.replace(hour=hour, minute=minute),
                "User": user,
                "Message": body,
            }
            continue

        if pending is not None:
            pending["Message"] += ("\n" if pending["Message"] else "") + line

    flush()
    return pd.DataFrame(records, columns=REQUIRED_COLUMNS)


def load_dataframe(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    text = decode_upload(raw)
    name = uploaded_file.name.lower()

    if name.endswith(".txt") or is_kakaotalk_text(text):
        parsed = parse_kakaotalk(text)
        if not parsed.empty:
            df = parsed
        else:
            df = None
    else:
        df = None

    if df is None:
        df = pd.read_csv(io.BytesIO(raw))
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                "Date, User, Message 칼럼이 필요합니다. "
                f"현재 칼럼: {list(df.columns)}"
            )
        df = df[REQUIRED_COLUMNS].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["User"] = df["User"].astype(str)
    df["Message"] = df["Message"].fillna("").astype(str)
    return df


def shorten_name(name: str, limit: int = 16) -> str:
    name = str(name)
    return name if len(name) <= limit else name[: limit - 1] + "…"


def participant_stats(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["글자 수"] = work["Message"].fillna("").astype(str).str.len()
    stats = (
        work.groupby("User", as_index=False)
        .agg(
            메시지_개수=("Message", "count"),
            총_글자_수=("글자 수", "sum"),
            평균_메시지_길이=("글자 수", "mean"),
        )
        .sort_values("메시지_개수", ascending=False)
        .reset_index(drop=True)
    )
    stats["평균_메시지_길이"] = stats["평균_메시지_길이"].round(1)
    stats["표시이름"] = stats["User"].map(shorten_name)
    return stats


def longest_message_row(df: pd.DataFrame) -> pd.Series:
    work = df.copy()
    work["글자 수"] = work["Message"].fillna("").astype(str).str.len()
    return work.loc[work["글자 수"].idxmax()]


def style_bar_chart(
    frame: pd.DataFrame,
    value_col: str,
    title: str,
    colorscale: list[str],
    hover_label: str,
    unit: str,
    decimals: int = 0,
):
    chart_df = frame.sort_values(value_col, ascending=True)
    value_format = f",.{decimals}f"
    fig = px.bar(
        chart_df,
        x=value_col,
        y="표시이름",
        orientation="h",
        color=value_col,
        color_continuous_scale=colorscale,
        custom_data=["User"],
        title=title,
    )
    fig.update_traces(
        marker_line_width=0,
        marker_cornerradius=6,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            f"{hover_label}: %{{x:{value_format}}}{unit}<extra></extra>"
        ),
    )
    fig.update_layout(
        height=max(CHART_MIN_HEIGHT, len(chart_df) * CHART_HEIGHT_PER_ROW + 80),
        plot_bgcolor="rgba(248,250,252,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif", size=13),
        title=dict(font=dict(size=18, color="#1f2937"), x=0, xanchor="left"),
        coloraxis_showscale=False,
        xaxis=dict(
            title=hover_label,
            gridcolor="#e5e7eb",
            zeroline=False,
            tickformat=",.0f" if decimals == 0 else ",.1f",
        ),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=10, r=24, t=56, b=16),
        bargap=0.28,
    )
    return fig


DASHBOARD_ACCENTS = [
    "#F59E0B",
    "#06B6D4",
    "#8B5CF6",
    "#10B981",
    "#F43F5E",
    "#3B82F6",
]


def inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetric"] label {
            color: #64748b !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 800 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def dashboard_period(df: pd.DataFrame) -> tuple[str, str, int, float]:
    dated = with_datetime(df)
    if dated.empty:
        return "기간 정보 없음", "-", 1, float(len(df))
    start = dated["Date"].min()
    end = dated["Date"].max()
    days = max(int((end.normalize() - start.normalize()).days) + 1, 1)
    daily_avg = len(df) / days
    return (
        f"{start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}",
        f"{days}일",
        days,
        daily_avg,
    )


def render_basic_stats(df: pd.DataFrame) -> None:
    inject_dashboard_css()
    stats = participant_stats(df)
    longest = longest_message_row(df)
    participants = df["User"].dropna().astype(str).unique().tolist()
    avg_length = (
        df["Message"].fillna("").astype(str).str.len().mean() if len(df) else 0
    )
    period_range, period_days, _days, daily_avg = dashboard_period(df)
    top_user = stats.iloc[0]["User"] if not stats.empty else "-"
    top_count = int(stats.iloc[0]["메시지_개수"]) if not stats.empty else 0

    st.subheader("📌 핵심 지표")
    row1 = st.columns(3)
    row1[0].metric("전체 메시지 수", f"{len(df):,}")
    row1[1].metric("대화 참여자 수", f"{len(participants):,}")
    row1[2].metric("대화 기간", period_days, period_range, delta_color="off")

    row2 = st.columns(3)
    row2[0].metric("일평균 메시지 수", f"{daily_avg:.1f}개")
    row2[1].metric("평균 메시지 길이", f"{avg_length:.1f}자")
    row2[2].metric("가장 많이 보낸 사람", str(top_user), f"{top_count:,}개", delta_color="off")

    st.subheader("👥 참여자별 통계")
    card_cols = st.columns(3)
    for index, row in stats.iterrows():
        accent = DASHBOARD_ACCENTS[int(index) % len(DASHBOARD_ACCENTS)]
        share = (row["메시지_개수"] / len(df) * 100) if len(df) else 0
        with card_cols[int(index) % 3]:
            st.markdown(
                f"<div style='height:6px;border-radius:999px;background:{accent};"
                f"margin:4px 0 10px 0;'></div>",
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                st.markdown(f"**{row['User']}**")
                m1, m2 = st.columns(2)
                m1.metric("메시지", f"{int(row['메시지_개수']):,}")
                m2.metric("평균 길이", f"{row['평균_메시지_길이']:.1f}자")
                st.caption(f"점유율 {share:.1f}% · 총 {int(row['총_글자_수']):,}자")

    st.subheader("📊 참여자별 메시지 개수")
    st.plotly_chart(
        style_bar_chart(
            stats,
            "메시지_개수",
            "참여자별 메시지 개수",
            ["#FDE68A", "#F59E0B", "#B45309"],
            "메시지 개수",
            "개",
        ),
        width="stretch",
    )

    st.subheader("✏️ 참여자별 평균 메시지 길이")
    st.plotly_chart(
        style_bar_chart(
            stats,
            "평균_메시지_길이",
            "참여자별 평균 메시지 길이",
            ["#C4B5FD", "#8B5CF6", "#6D28D9"],
            "평균 메시지 길이",
            "자",
            decimals=1,
        ),
        width="stretch",
    )

    st.subheader("🏆 가장 긴 메시지")
    date_text = (
        pd.to_datetime(longest["Date"]).strftime("%Y-%m-%d %H:%M")
        if pd.notna(longest["Date"])
        else "-"
    )
    highlight = st.columns(3)
    highlight[0].metric("보낸 사람", str(longest["User"]))
    highlight[1].metric("글자 수", f"{int(longest['글자 수']):,}자")
    highlight[2].metric("보낸 시각", date_text)
    safe_message = html.escape(str(longest["Message"]))
    st.markdown(
        f"<div style='background:#fffbeb;border:1px solid #fcd34d;"
        f"border-radius:16px;padding:16px 18px;line-height:1.6;"
        f"white-space:pre-wrap;box-shadow:0 8px 20px rgba(245,158,11,0.08);'>"
        f"{safe_message}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("👀 데이터 미리보기"):
        st.dataframe(df.head(10), width="stretch")


WEEKDAYS_KR = ["월", "화", "수", "목", "금", "토", "일"]
CHART_FONT = dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif", size=13)


def with_datetime(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    return work.dropna(subset=["Date"]).copy()


def hour_distribution(work: pd.DataFrame) -> pd.DataFrame:
    counts = (
        work.assign(시간=work["Date"].dt.hour)
        .groupby("시간")
        .size()
        .reindex(range(24), fill_value=0)
        .rename("메시지_개수")
        .reset_index()
    )
    counts["시간대"] = counts["시간"].astype(int).astype(str) + "시"
    return counts


def weekday_distribution(work: pd.DataFrame) -> pd.DataFrame:
    counts = (
        work.assign(요일번호=work["Date"].dt.dayofweek)
        .groupby("요일번호")
        .size()
        .reindex(range(7), fill_value=0)
        .rename("메시지_개수")
        .reset_index()
    )
    counts["요일"] = counts["요일번호"].map(lambda i: WEEKDAYS_KR[int(i)])
    counts["요일"] = pd.Categorical(counts["요일"], categories=WEEKDAYS_KR, ordered=True)
    return counts.sort_values("요일")


def monthly_trend(work: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        work.set_index("Date")
        .resample("MS")
        .size()
        .rename("메시지_개수")
        .reset_index()
    )
    monthly["월"] = monthly["Date"].dt.strftime("%Y년 %m월")
    return monthly


def peak_labels(frame: pd.DataFrame, label_col: str, value_col: str = "메시지_개수") -> tuple[str, int]:
    peak_value = int(frame[value_col].max())
    labels = frame.loc[frame[value_col] == peak_value, label_col].astype(str).tolist()
    return ", ".join(labels), peak_value


def apply_time_chart_layout(fig, y_title: str = "메시지 개수") :
    fig.update_layout(
        height=440,
        plot_bgcolor="rgba(248,250,252,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=CHART_FONT,
        title=dict(font=dict(size=18, color="#1f2937"), x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="", gridcolor="#e5e7eb", zeroline=False),
        yaxis=dict(title=y_title, gridcolor="#e5e7eb", zeroline=False),
        margin=dict(l=10, r=24, t=72, b=16),
        hovermode="x unified",
    )
    return fig


def hour_bar_chart(hour_df: pd.DataFrame, peak_hours: str):
    peak_set = set(peak_hours.split(", "))
    chart_df = hour_df.copy()
    chart_df["구분"] = chart_df["시간대"].map(
        lambda label: "가장 활발한 시간대" if label in peak_set else "그 외"
    )
    fig = px.bar(
        chart_df,
        x="시간대",
        y="메시지_개수",
        color="구분",
        color_discrete_map={"가장 활발한 시간대": "#059669", "그 외": "#6EE7B7"},
        title="시간대별 메시지 분포",
        category_orders={"시간대": [f"{h}시" for h in range(24)]},
    )
    fig.update_traces(
        marker_line_width=0,
        marker_cornerradius=4,
        hovertemplate="<b>%{x}</b><br>메시지 %{y:,}개<extra></extra>",
    )
    return apply_time_chart_layout(fig)


def weekday_bar_chart(weekday_df: pd.DataFrame, peak_days: str):
    peak_set = set(peak_days.split(", "))
    chart_df = weekday_df.copy()
    chart_df["구분"] = chart_df["요일"].astype(str).map(
        lambda label: "가장 활발한 요일" if label in peak_set else "그 외"
    )
    fig = px.bar(
        chart_df,
        x="요일",
        y="메시지_개수",
        color="구분",
        color_discrete_map={"가장 활발한 요일": "#DB2777", "그 외": "#F9A8D4"},
        title="요일별 메시지 분포",
        category_orders={"요일": WEEKDAYS_KR},
    )
    fig.update_traces(
        marker_line_width=0,
        marker_cornerradius=8,
        hovertemplate="<b>%{x}요일</b><br>메시지 %{y:,}개<extra></extra>",
    )
    return apply_time_chart_layout(fig)


def monthly_line_chart(monthly_df: pd.DataFrame):
    fig = px.line(
        monthly_df,
        x="Date",
        y="메시지_개수",
        markers=True,
        custom_data=["월"],
        title="월별 메시지 추이",
    )
    fig.update_traces(
        line=dict(color="#2563EB", width=3, shape="spline"),
        marker=dict(size=9, color="#1D4ED8", line=dict(width=2, color="white")),
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.12)",
        hovertemplate="<b>%{customdata[0]}</b><br>메시지 %{y:,}개<extra></extra>",
    )
    fig.update_xaxes(tickformat="%Y년 %m월")
    return apply_time_chart_layout(fig)


def render_time_analysis(df: pd.DataFrame) -> None:
    work = with_datetime(df)
    if work.empty:
        st.warning("날짜를 읽을 수 있는 메시지가 없어 시간대 분석을 할 수 없습니다.")
        return

    hour_df = hour_distribution(work)
    weekday_df = weekday_distribution(work)
    monthly_df = monthly_trend(work)

    peak_hours, peak_hour_count = peak_labels(hour_df, "시간대")
    peak_days, peak_day_count = peak_labels(weekday_df, "요일")
    peak_day_display = ", ".join(f"{name}요일" for name in peak_days.split(", "))

    st.subheader("🔥 가장 활발한 시간")
    col1, col2 = st.columns(2)
    col1.metric("가장 활발한 시간대", peak_hours, f"{peak_hour_count:,}개")
    col2.metric("가장 활발한 요일", peak_day_display, f"{peak_day_count:,}개")

    st.subheader("🕐 시간대별 메시지")
    st.plotly_chart(hour_bar_chart(hour_df, peak_hours), width="stretch")

    st.subheader("📅 요일별 메시지")
    st.plotly_chart(weekday_bar_chart(weekday_df, peak_days), width="stretch")

    st.subheader("📈 월별 메시지 추이")
    st.plotly_chart(monthly_line_chart(monthly_df), width="stretch")


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TOKEN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{2,}")
NUMERIC_ONLY_RE = re.compile(r"^[\d\s:.\-]+$")
SYSTEM_MESSAGE_RE = re.compile(
    r"^(사진(\s*\d+\s*장)?|이모티콘|동영상|보이스톡.*|페이스톡.*|삭제된 메시지)$"
)
SYSTEM_WORDS = {"사진", "이모티콘", "동영상", "보이스톡", "페이스톡"}
LAUGH_ONLY_RE = re.compile(r"^[ㅋㅎ]+$")


def korean_font_path() -> str | None:
    system = platform.system()
    if system == "Windows":
        candidates = [
            r"C:\Windows\Fonts\malgun.ttf",
            r"C:\Windows\Fonts\malgunbd.ttf",
        ]
    elif system == "Darwin":
        candidates = [
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/System/Library/Fonts/AppleGothic.ttf",
            "/Library/Fonts/AppleGothic.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def is_excluded_message(text: str) -> bool:
    text = str(text).strip()
    if not text:
        return True
    if SYSTEM_MESSAGE_RE.match(text):
        return True
    without_url = URL_RE.sub("", text).strip()
    if not without_url:
        return True
    if NUMERIC_ONLY_RE.fullmatch(without_url):
        return True
    return False


def tokenize_message(text: str) -> list[str]:
    cleaned = URL_RE.sub(" ", str(text))
    tokens: list[str] = []
    for token in TOKEN_RE.findall(cleaned):
        if len(token) < 2:
            continue
        if token in SYSTEM_WORDS:
            continue
        if LAUGH_ONLY_RE.fullmatch(token):
            continue
        tokens.append(token)
    return tokens


def word_counters(df: pd.DataFrame) -> tuple[Counter, dict[str, Counter]]:
    overall: Counter = Counter()
    by_user: dict[str, Counter] = defaultdict(Counter)
    for user, message in zip(df["User"], df["Message"]):
        if is_excluded_message(message):
            continue
        tokens = tokenize_message(message)
        overall.update(tokens)
        by_user[str(user)].update(tokens)
    return overall, dict(by_user)


def laugh_counts(df: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    work = df.copy()
    work["ㅋ"] = work["Message"].fillna("").astype(str).str.contains("ㅋ")
    work["ㅎ"] = work["Message"].fillna("").astype(str).str.contains("ㅎ")
    overall = {"ㅋ": int(work["ㅋ"].sum()), "ㅎ": int(work["ㅎ"].sum())}
    by_user = (
        work.groupby("User", as_index=False)[["ㅋ", "ㅎ"]]
        .sum()
        .sort_values(["ㅋ", "ㅎ"], ascending=False)
    )
    by_user["표시이름"] = by_user["User"].map(shorten_name)
    return overall, by_user


def word_bar_chart(items: list[tuple[str, int]], title: str, colorscale: list[str]):
    frame = pd.DataFrame(items, columns=["단어", "횟수"]).sort_values("횟수", ascending=True)
    fig = px.bar(
        frame,
        x="횟수",
        y="단어",
        orientation="h",
        color="횟수",
        color_continuous_scale=colorscale,
        title=title,
    )
    fig.update_traces(
        marker_line_width=0,
        marker_cornerradius=6,
        hovertemplate="<b>%{y}</b><br>%{x:,}회<extra></extra>",
    )
    fig.update_layout(
        height=max(CHART_MIN_HEIGHT, len(frame) * CHART_HEIGHT_PER_ROW + 80),
        plot_bgcolor="rgba(248,250,252,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=CHART_FONT,
        title=dict(font=dict(size=18, color="#1f2937"), x=0, xanchor="left"),
        coloraxis_showscale=False,
        xaxis=dict(title="횟수", gridcolor="#e5e7eb", zeroline=False),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=10, r=24, t=56, b=16),
        bargap=0.28,
    )
    return fig


def laugh_compare_chart(by_user: pd.DataFrame):
    melted = by_user.melt(
        id_vars=["User", "표시이름"],
        value_vars=["ㅋ", "ㅎ"],
        var_name="종류",
        value_name="횟수",
    )
    melted["종류"] = melted["종류"].map({"ㅋ": "ㅋ 포함", "ㅎ": "ㅎ 포함"})
    order = by_user.sort_values(["ㅋ", "ㅎ"]).loc[:, "표시이름"].tolist()
    fig = px.bar(
        melted,
        x="횟수",
        y="표시이름",
        orientation="h",
        color="종류",
        barmode="group",
        custom_data=["User", "종류"],
        color_discrete_map={"ㅋ 포함": "#F59E0B", "ㅎ 포함": "#38BDF8"},
        title="참여자별 ㅋ / ㅎ 포함 메시지 횟수",
        category_orders={"표시이름": order},
    )
    fig.update_traces(
        marker_line_width=0,
        marker_cornerradius=4,
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}: %{x:,}회<extra></extra>",
    )
    fig.update_layout(
        height=max(CHART_MIN_HEIGHT, len(by_user) * 32 + 80),
        plot_bgcolor="rgba(248,250,252,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=CHART_FONT,
        title=dict(font=dict(size=18, color="#1f2937"), x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=""),
        xaxis=dict(title="메시지 횟수", gridcolor="#e5e7eb", zeroline=False),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=10, r=24, t=72, b=16),
        bargap=0.28,
    )
    return fig


def make_wordcloud_figure(freq: Counter, font_path: str):
    cloud = WordCloud(
        font_path=font_path,
        width=1400,
        height=700,
        background_color="white",
        colormap="autumn",
        max_words=200,
        prefer_horizontal=0.85,
        min_font_size=10,
        collocations=False,
    ).generate_from_frequencies(dict(freq))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(cloud, interpolation="bilinear")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    return fig


def render_word_analysis(df: pd.DataFrame) -> None:
    overall, by_user = word_counters(df)
    laugh_overall, laugh_by_user = laugh_counts(df)
    font_path = korean_font_path()

    st.subheader("😄 ㅋ / ㅎ 한눈에 보기")
    col1, col2 = st.columns(2)
    col1.metric("ㅋ 포함 메시지", f"{laugh_overall['ㅋ']:,}회")
    col2.metric("ㅎ 포함 메시지", f"{laugh_overall['ㅎ']:,}회")

    if not overall:
        st.warning("분석할 단어가 없습니다. 필터 조건을 확인해 주세요.")
        st.plotly_chart(laugh_compare_chart(laugh_by_user), width="stretch")
        return

    st.subheader("🔤 자주 쓰는 단어 TOP 20")
    top20 = overall.most_common(20)
    st.plotly_chart(
        word_bar_chart(top20, "가장 많이 쓴 단어 TOP 20", ["#FDE68A", "#F59E0B", "#B45309"]),
        width="stretch",
    )

    st.subheader("👤 참여자별 단어 TOP 10")
    users_with_words = [user for user, counter in by_user.items() if counter]
    if not users_with_words:
        st.info("참여자별 단어를 집계하지 못했습니다.")
    else:
        selected = st.selectbox("참여자 선택", users_with_words)
        user_top = by_user[selected].most_common(10)
        st.plotly_chart(
            word_bar_chart(
                user_top,
                f"{selected} TOP 10",
                ["#C4B5FD", "#8B5CF6", "#6D28D9"],
            ),
            width="stretch",
        )

    st.subheader("☁️ 워드클라우드")
    if not font_path:
        st.warning(
            "한글 폰트를 찾지 못해 워드클라우드를 만들 수 없습니다. "
            "Windows는 malgun.ttf, Mac은 AppleGothic이 필요합니다."
        )
    else:
        fig = make_wordcloud_figure(overall, font_path)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.subheader("😆 ㅋ / ㅎ 포함 메시지 비교")
    st.plotly_chart(laugh_compare_chart(laugh_by_user), width="stretch")


def render_sidebar_intro() -> object:
    with st.sidebar:
        st.markdown("## 💬 카카오톡 대화 분석기")
        st.caption(
            "카카오톡 대화를 올려 누가, 언제, 어떤 말을 많이 했는지 "
            "한눈에 살펴보는 분석 앱입니다."
        )
        st.markdown("---")
        return st.file_uploader(
            "📂 CSV 파일 업로드",
            type=["csv", "txt"],
            help="Date, User, Message 칼럼 CSV 또는 카카오톡 내보내기 txt",
        )


def render_sidebar_details(df: pd.DataFrame | None) -> None:
    with st.sidebar:
        if df is not None and not df.empty:
            dated = with_datetime(df)
            participants = df["User"].dropna().astype(str).unique().tolist()
            if dated.empty:
                period = "기간 정보 없음"
            else:
                start = dated["Date"].min().strftime("%Y-%m-%d")
                end = dated["Date"].max().strftime("%Y-%m-%d")
                period = f"{start} ~ {end}"

            st.markdown("### 📌 기본 정보")
            st.metric("메시지 수", f"{len(df):,}")
            st.markdown(f"**기간**  \n{period}")
            st.markdown("**참여자**")
            st.write(", ".join(participants))

        st.markdown("---")
        st.markdown("### 📖 사용 방법")
        st.markdown(
            """
1. 위에서 **CSV** 또는 카카오톡 **txt** 파일을 업로드하세요.
2. **📊 기본 통계**에서 메시지 수와 참여자별 현황을 확인하세요.
3. **⏰ 시간 분석**에서 시간대·요일·월별 흐름을 보세요.
4. **💬 단어 분석**에서 자주 쓴 단어와 워드클라우드를 확인하세요.
            """
        )


def main() -> None:
    st.set_page_config(page_title="카카오톡 대화 분석기", layout="wide")
    st.title("💬 카카오톡 대화 분석기")

    uploaded_file = render_sidebar_intro()
    df: pd.DataFrame | None = None

    if uploaded_file is not None:
        try:
            loaded = load_dataframe(uploaded_file)
        except Exception as exc:
            st.error(f"파일을 불러오지 못했습니다: {exc}")
            render_sidebar_details(None)
            return
        if loaded.empty:
            st.warning("메시지가 없습니다. 파일 형식을 확인해 주세요.")
            render_sidebar_details(None)
            return
        df = loaded

    render_sidebar_details(df)

    if df is None:
        st.info("왼쪽 사이드바에서 파일을 업로드하면 분석 탭이 나타납니다.")
        return

    tab_basic, tab_time, tab_word = st.tabs(
        ["📊 기본 통계", "⏰ 시간 분석", "💬 단어 분석"]
    )
    with tab_basic:
        render_basic_stats(df)
    with tab_time:
        render_time_analysis(df)
    with tab_word:
        render_word_analysis(df)


if __name__ == "__main__":
    main()
