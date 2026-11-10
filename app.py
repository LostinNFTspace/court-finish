from __future__ import annotations

from datetime import datetime, time
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


YEARS = range(2018, 2025)
SIMULATIONS = 2_000
MIN_GROUP_SIZE = 200
CHANGEOVER_DEFAULT = 15
RIGA_TIMEZONE = ZoneInfo("Europe/Riga")
ROUND_OPTIONS = ["R64", "R32", "R16", "QF", "SF", "F"]
PRIMARY_URL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
    "atp_matches_qual_chall_{year}.csv"
)
ARCHIVE_URL = (
    "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/"
    "main/atp/atp_matches_qual_chall_{year}.csv"
)
REQUIRED_COLUMNS = {
    "surface",
    "round",
    "best_of",
    "minutes",
    "score",
    "tourney_level",
    "tourney_name",
    "tourney_date",
    "winner_name",
    "loser_name",
}


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_matches() -> tuple[pd.DataFrame, dict[int, dict[str, str]]]:
    """Fetch the seven requested seasons, preferring Jeff Sackmann's source."""
    frames: list[pd.DataFrame] = []
    sources: dict[int, dict[str, str]] = {}

    for year in YEARS:
        loaded = False
        urls = [
            (PRIMARY_URL.format(year=year), "Jeff Sackmann's tennis_atp"),
            (ARCHIVE_URL.format(year=year), "archived copy"),
        ]
        for url, source_name in urls:
            try:
                request = Request(
                    url,
                    headers={"User-Agent": "CourtFinish/1.0 (historical tennis data)"},
                )
                with urlopen(request, timeout=30) as response:
                    csv_bytes = response.read()
                frame = pd.read_csv(BytesIO(csv_bytes))
                missing = REQUIRED_COLUMNS - set(frame.columns)
                if missing:
                    raise ValueError(
                        "CSV is missing required columns: "
                        + ", ".join(sorted(missing))
                    )
                frame["source_year"] = year
                frames.append(frame)
                sources[year] = {"source": source_name, "url": url}
                loaded = True
                break
            except (HTTPError, URLError, TimeoutError, OSError, ValueError):
                continue

        if not loaded:
            raise RuntimeError(
                f"Could not load the {year} match file from the original source "
                "or the archival copy."
            )

    if len(frames) != len(YEARS):
        raise RuntimeError("The complete 2018–2024 dataset could not be loaded.")
    return pd.concat(frames, ignore_index=True), sources


def clean_matches(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep Challenger main-draw matches and apply disjoint cleaning reasons."""
    data = raw.copy()
    data["minutes"] = pd.to_numeric(data["minutes"], errors="coerce")
    data["best_of"] = pd.to_numeric(data["best_of"], errors="coerce")

    excluded: dict[str, int] = {}
    remaining = pd.Series(True, index=data.index)

    round_values = data["round"].fillna("").astype(str).str.strip()
    challenger_main_draw = (
        data["tourney_level"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("C")
        & ~round_values.str.fullmatch(
            r"(?:Q\d+|Qualifying(?: Round)?)", case=False
        )
    )
    outside_challenger_main_draw = ~challenger_main_draw & remaining
    excluded[
        "Not Challenger main draw (ATP/Grand Slam/qualifying)"
    ] = int(outside_challenger_main_draw.sum())
    remaining &= challenger_main_draw

    missing_minutes = data["minutes"].isna() & remaining
    excluded["Missing match duration"] = int(missing_minutes.sum())
    remaining &= ~missing_minutes

    interrupted = (
        data["score"]
        .fillna("")
        .astype(str)
        .str.contains(r"RET|W/O|DEF", case=False, regex=True)
        & remaining
    )
    excluded["Retirement, walkover, or default"] = int(interrupted.sum())
    remaining &= ~interrupted

    best_of_three = data["best_of"].eq(3)
    impossible = (
        best_of_three
        & ((data["minutes"] < 30) | (data["minutes"] > 300))
        & remaining
    )
    excluded["Impossible best-of-3 duration (<30 or >300 min)"] = int(
        impossible.sum()
    )
    remaining &= ~impossible

    other_format = data["best_of"].notna() & ~best_of_three & remaining
    excluded["Not best-of-3"] = int(other_format.sum())
    remaining &= ~other_format

    missing_model_fields = (
        data["surface"].isna() | data["round"].isna() | data["best_of"].isna()
    ) & remaining
    excluded["Missing surface, round, or match format"] = int(
        missing_model_fields.sum()
    )
    remaining &= ~missing_model_fields

    clean = data.loc[remaining].copy()
    clean["surface"] = clean["surface"].astype(str).str.strip()
    clean["round"] = clean["round"].astype(str).str.strip()
    return clean, excluded


def duration_sample(
    clean: pd.DataFrame, surface: str, round_name: str
) -> tuple[np.ndarray, str]:
    exact = clean.loc[
        clean["surface"].eq(surface) & clean["round"].eq(round_name), "minutes"
    ].to_numpy(dtype=float)
    if len(exact) >= MIN_GROUP_SIZE:
        return exact, "Surface + round"

    surface_only = clean.loc[clean["surface"].eq(surface), "minutes"].to_numpy(
        dtype=float
    )
    if len(surface_only) == 0:
        raise ValueError(f"No usable historical matches were found for {surface}.")
    return surface_only, "Surface fallback"


def clock_minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def parse_clock(value: object) -> time | None:
    """'19:00', '19.00', '1900' or '19' -> time; empty -> None (no restriction)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, time):
        return value
    text = str(value).strip().replace(".", ":").replace(";", ":")
    if not text:
        return None
    if ":" not in text:
        text = f"{text}:00" if len(text) <= 2 else f"{text[:-2]}:{text[-2:]}"
    try:
        hour, minute = (int(part) for part in text.split(":")[:2])
        return time(hour, minute)
    except ValueError:
        st.warning(f"Couldn't read \"{value}\" as a time, so that match has no earliest start. Use e.g. 19:00.")
        return None


def format_clock(total_minutes: float | int) -> str:
    minute_value = max(0, int(round(total_minutes)))
    day_offset, within_day = divmod(minute_value, 24 * 60)
    hour, minute = divmod(within_day, 60)
    suffix = f" (+{day_offset} day)" if day_offset == 1 else (
        f" (+{day_offset} days)" if day_offset > 1 else ""
    )
    return f"{hour:02d}:{minute:02d}{suffix}"


def quantile(values: np.ndarray, percentile: int) -> int:
    return int(round(float(np.percentile(values, percentile))))


def schedule_forecast(
    clean: pd.DataFrame,
    surface: str,
    first_start: time,
    changeover: int,
    schedule: list[dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    rng = np.random.default_rng(2026)
    available_at = np.full(SIMULATIONS, clock_minutes(first_start), dtype=float)
    forecast_rows: list[dict[str, object]] = []
    chart_rows: list[dict[str, object]] = []
    duration_stats: dict[str, int] = {}
    final_end = np.zeros(SIMULATIONS, dtype=float)

    for index, match in enumerate(schedule, start=1):
        round_label = str(match["round"])
        sample, sample_source = duration_sample(clean, surface, round_label)
        round_minutes = np.asarray(sample, dtype=float)
        duration_stats[round_label] = len(round_minutes)
        durations = rng.choice(round_minutes, size=SIMULATIONS, replace=True)

        not_before = match.get("not_before")
        not_before_minutes = (
            clock_minutes(not_before) if isinstance(not_before, time) else 0
        )
        starts = np.maximum(available_at, not_before_minutes)
        ends = starts + durations

        match_name = str(match["match"])
        duration_p10 = quantile(round_minutes, 10)
        duration_p50 = quantile(round_minutes, 50)
        duration_p90 = quantile(round_minutes, 90)
        start_p10 = quantile(starts, 10)
        start_p50 = quantile(starts, 50)
        start_p90 = quantile(starts, 90)
        end_p10 = quantile(ends, 10)
        end_p50 = quantile(ends, 50)
        end_p90 = quantile(ends, 90)

        forecast_rows.append(
            {
                "Match": match_name,
                "Round": round_label,
                "Sample": sample_source,
                "Historical matches": len(round_minutes),
                "Duration P10": f"{duration_p10} min",
                "Duration median": f"{duration_p50} min",
                "Duration P90": f"{duration_p90} min",
                "Start P10": format_clock(start_p10),
                "Start likely": format_clock(start_p50),
                "Start P90": format_clock(start_p90),
                "Finish P10": format_clock(end_p10),
                "Finish likely": format_clock(end_p50),
                "Finish P90": format_clock(end_p90),
            }
        )
        chart_rows.append(
            {
                "Match": match_name,
                "Round": round_label,
                "range_start": start_p10,
                "range_finish": end_p90,
                "median_start": start_p50,
                "median_finish": end_p50,
                "Earliest start": format_clock(start_p10),
                "Latest finish": format_clock(end_p90),
                "Likely start": format_clock(start_p50),
                "Likely finish": format_clock(end_p50),
            }
        )

        available_at = ends + changeover
        final_end = ends

    finish_quantiles = {
        "early": quantile(final_end, 10),
        "likely": quantile(final_end, 50),
        "late": quantile(final_end, 90),
    }
    return (
        pd.DataFrame(forecast_rows),
        finish_quantiles,
        pd.DataFrame(chart_rows),
        final_end,
    )


def historical_duration_table(clean: pd.DataFrame, surface: str) -> pd.DataFrame:
    rows = []
    for round_label in ROUND_OPTIONS:
        sample, source = duration_sample(clean, surface, round_label)
        rows.append(
            {
                "Round": round_label,
                "Sample used": len(sample),
                "Estimate based on": source,
                "10th percentile": f"{quantile(sample, 10)} min",
                "Median": f"{quantile(sample, 50)} min",
                "90th percentile": f"{quantile(sample, 90)} min",
            }
        )
    return pd.DataFrame(rows)


def build_timeline(chart_data: pd.DataFrame) -> alt.LayerChart:
    min_time = int(chart_data["range_start"].min())
    max_time = int(chart_data["range_finish"].max())
    axis_start = (min_time // 60) * 60
    axis_end = ((max_time + 59) // 60) * 60
    if axis_end <= axis_start:
        axis_end = axis_start + 60

    clock_label = (
        "(floor(datum.value / 60) % 24 < 10 ? '0' : '') + "
        "(floor(datum.value / 60) % 24) + ':' + "
        "(datum.value % 60 < 10 ? '0' : '') + (datum.value % 60) + "
        "(datum.value >= 1440 ? ' +1d' : '')"
    )
    shared_x = alt.X(
        "range_start:Q",
        scale=alt.Scale(domain=[axis_start, axis_end], nice=False),
        axis=alt.Axis(
            title="Local court time",
            values=list(range(axis_start, axis_end + 1, 60)),
            labelExpr=clock_label,
            grid=True,
        ),
    )
    band = (
        alt.Chart(chart_data)
        .mark_bar(height=18, cornerRadius=5, color="#b8d7cc")
        .encode(
            x=shared_x,
            x2="range_finish:Q",
            y=alt.Y("Match:N", sort=None, title=None),
            tooltip=[
                alt.Tooltip("Match:N"),
                alt.Tooltip("Round:N"),
                alt.Tooltip("Earliest start:N", title="Earliest likely start"),
                alt.Tooltip("Latest finish:N", title="Latest likely finish"),
            ],
        )
    )
    median = (
        alt.Chart(chart_data)
        .mark_bar(height=8, cornerRadius=4, color="#176c59")
        .encode(
            x="median_start:Q",
            x2="median_finish:Q",
            y=alt.Y("Match:N", sort=None, title=None),
            tooltip=[
                alt.Tooltip("Match:N"),
                alt.Tooltip("Round:N"),
                alt.Tooltip("Likely start:N"),
                alt.Tooltip("Likely finish:N"),
            ],
        )
    )
    return (band + median).properties(height=max(190, 48 * len(chart_data)))


st.set_page_config(
    page_title="When will the court finish?",
    page_icon="🎾",
    layout="wide",
)

st.title("When will the last match on this court finish?")
st.markdown(
    "Tennis has no fixed end time, so crews and broadcasts plan blind. "
    "This predicts the court's finish from **thousands of real ATP Challenger matches (2018–2024)**, "
    "so you know when the crew is done and whether overtime is coming."
)
hero = st.container()
plan_box = st.container()
details_box = st.container()
st.divider()
audit_box = st.container()

try:
    with st.spinner("Loading ATP Challenger match history for 2018–2024…"):
        raw_matches, source_by_year = load_matches()
        matches, exclusion_counts = clean_matches(raw_matches)
except Exception as error:
    st.error(f"Historical data could not be loaded. {error}")
    st.caption(
        "The app needs the seven public match files for 2018–2024. "
        "Check the source links or retry when GitHub is available."
    )
    st.stop()

archive_years = [
    year
    for year, source in source_by_year.items()
    if source["source"] == "archived copy"
]
if archive_years:
    audit_box.info(
        "The data comes from a third-party GitHub copy of Jeff Sackmann’s "
        "dataset (github.com/Aneeshers/tennis-sackmann-archive), because the "
        "original file links returned 404."
    )

total_excluded = sum(exclusion_counts.values())
used_count = len(matches)
with audit_box.expander("Where the data comes from, and how it was cleaned", expanded=False):
    audit_cols = st.columns(3)
    audit_cols[0].metric("Source rows", f"{len(raw_matches):,}")
    audit_cols[1].metric("Matches used", f"{used_count:,}")
    audit_cols[2].metric("Excluded", f"{total_excluded:,}")

    audit_table = pd.DataFrame(
        [
            {"Cleaning result": "Used in duration model", "Matches": used_count},
            *[
                {"Cleaning result": reason, "Matches": count}
                for reason, count in exclusion_counts.items()
            ],
        ]
    )
    st.dataframe(audit_table, hide_index=True, width="stretch")
    raw_by_year = raw_matches.groupby("source_year").size()
    used_by_year = matches.groupby("source_year").size()
    yearly_counts = pd.DataFrame(
        [
            {
                "Year": year,
                "Downloaded rows": int(raw_by_year.get(year, 0)),
                "Matches used": int(used_by_year.get(year, 0)),
                "Excluded": int(raw_by_year.get(year, 0) - used_by_year.get(year, 0)),
            }
            for year in YEARS
        ]
    )
    st.markdown("**Matches by year**")
    st.dataframe(yearly_counts, hide_index=True, width="stretch")

    normalized_rounds = raw_matches["round"].fillna("").astype(str).str.strip()
    source_round_counts = (
        raw_matches.assign(_audit_round=normalized_rounds)
        .groupby(["source_year", "_audit_round"])
        .size()
    )
    used_rounds = matches["round"].fillna("").astype(str).str.strip()
    used_round_counts = (
        matches.assign(_audit_round=used_rounds)
        .groupby(["source_year", "_audit_round"])
        .size()
    )
    q_rounds_2019 = sum(
        int(source_round_counts.get((2019, round_name), 0))
        for round_name in ("Q1", "Q2", "Q3")
    )
    q_rounds_2024 = sum(
        int(source_round_counts.get((2024, round_name), 0))
        for round_name in ("Q1", "Q2", "Q3")
    )
    r64_2019 = int(source_round_counts.get((2019, "R64"), 0))
    used_r64_2019 = int(used_round_counts.get((2019, "R64"), 0))
    st.caption(
        f"Why 2019 keeps more: its file has {q_rounds_2019:,} rows labeled "
        f"Q1–Q3, compared with {q_rounds_2024:,} in 2024, and it includes "
        f"{r64_2019:,} R64 matches ({used_r64_2019:,} retained). Since the "
        "retention rate uses all downloaded rows as its denominator, these "
        "different round mixes raise 2019’s rate; the match-level cleaning "
        "rules are unchanged."
    )

    removed_rows = raw_matches.loc[~raw_matches.index.isin(matches.index)].copy()
    removed_rows["_audit_round"] = (
        removed_rows["round"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "Missing round")
    )
    removed_by_round = (
        removed_rows.groupby(["source_year", "_audit_round"])
        .size()
        .unstack(fill_value=0)
        .reindex(YEARS, fill_value=0)
    )
    round_order = ["Q1", "Q2", "Q3", "R64", "R32", "R16", "QF", "SF", "F"]
    ordered_rounds = [name for name in round_order if name in removed_by_round]
    ordered_rounds.extend(
        sorted(set(removed_by_round.columns) - set(ordered_rounds))
    )
    removed_by_round = removed_by_round.reindex(
        columns=ordered_rounds, fill_value=0
    ).astype(int)
    removed_by_round.index.name = "Year"
    st.markdown("**Removed rows by year and round**")
    st.dataframe(removed_by_round.reset_index(), hide_index=True, width="stretch")
    st.caption(
        "Counts include rows removed by any cleaning rule. Round labels are "
        "shown as provided in the source; Q1–Q3 are qualifying rounds."
    )

    st.markdown("**Exact download URLs**")
    st.code(
        "\n".join(
            f"{year}: {source_by_year[year]['url']}" for year in YEARS
        ),
        language=None,
    )

    sample = matches.sample(n=min(5, len(matches)), random_state=42).copy()
    sample["Date"] = pd.to_datetime(
        sample["tourney_date"].astype("string").str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d",
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    sample_table = sample[
        ["tourney_name", "Date", "winner_name", "loser_name", "score", "minutes"]
    ].rename(
        columns={
            "tourney_name": "Tournament",
            "winner_name": "Winner",
            "loser_name": "Loser",
            "score": "Score",
            "minutes": "Minutes",
        }
    )
    sample_table["Minutes"] = sample_table["Minutes"].round().astype(int)
    st.markdown("**Five random matches used in the model**")
    st.dataframe(sample_table, hide_index=True, width="stretch")

    st.caption(
        "Only Challenger main-draw matches are considered; ATP tour, Grand Slam, "
        "and qualifying rows are excluded. Retirements, walkovers, and defaults "
        "are identified from the score text."
    )
    st.markdown(
        "Data source: third-party GitHub copy of Jeff Sackmann’s dataset: "
        "[github.com/Aneeshers/tennis-sackmann-archive]"
        "(https://github.com/Aneeshers/tennis-sackmann-archive). "
        "The original file links returned 404."
    )

surfaces = sorted(matches["surface"].dropna().unique().tolist())
if not surfaces:
    st.error("No matches with a usable surface were found in the downloaded files.")
    st.stop()
surface_index = surfaces.index("Hard") if "Hard" in surfaces else 0

if "initial_schedule" not in st.session_state:
    st.session_state.initial_schedule = pd.DataFrame(
        {
            "Match": ["Match 1", "Match 2", "Match 3", "Match 4", "Match 5"],
            "Round": ["R32", "R32", "R32", "R32", "R32"],
            "Not before": ["", "", "", "19:00", ""],
        }
    )

plan_box.subheader("Change the court plan")
plan_box.caption("Edit the order of play below, then press Update forecast. "
                 "\"Not before\": leave empty, or type a time if a match can't start earlier (an evening session or a TV slot). "
                 "Match 4 in the sample is an evening match at 19:00, which is why the court waits.")
with plan_box.form("court_plan"):
    setting_cols = st.columns([1.2, 1, 1, 1])
    surface = setting_cols[0].selectbox(
        "Surface",
        options=surfaces,
        index=surface_index,
    )
    first_start = setting_cols[1].time_input(
        "First match start",
        value=time(11, 0),
        step=900,
        help="The earliest start for the first match on this court.",
    )
    changeover = setting_cols[2].number_input(
        "Changeover between matches (minutes)",
        min_value=0,
        max_value=60,
        value=CHANGEOVER_DEFAULT,
        step=5,
    )
    shift_end = setting_cols[3].time_input(
        "Crew shift ends",
        value=time(18, 30),
        step=900,
        help="Planned end of the crew's shift. Anything later is overtime.",
    )

    edited_schedule = st.data_editor(
        st.session_state.initial_schedule,
        key="schedule_editor",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "Match": st.column_config.TextColumn(
                "Match",
                help="A label for this match on the order of play.",
                required=False,
            ),
            "Round": st.column_config.SelectboxColumn(
                "Round",
                options=ROUND_OPTIONS,
                required=True,
            ),
            "Not before": st.column_config.TextColumn(
                "Not before (optional)",
                help="Leave empty: the match starts as soon as the court is free. "
                     "Type a time (e.g. 19:00) if it can't start earlier, such as an evening session or a TV slot.",
                required=False,
            ),
        },
    )
    forecast_submitted = st.form_submit_button(
        "Update forecast", type="primary"
    )

if forecast_submitted or "forecast_run_at" not in st.session_state:
    st.session_state.forecast_run_at = datetime.now(RIGA_TIMEZONE)

schedule: list[dict[str, object]] = []
for row_index, row in edited_schedule.reset_index(drop=True).iterrows():
    round_value = row.get("Round")
    if pd.isna(round_value) or not str(round_value).strip():
        continue
    label = row.get("Match")
    label = str(label).strip() if pd.notna(label) else ""
    schedule.append(
        {
            "match": label or f"Match {row_index + 1}",
            "round": str(round_value),
            "not_before": parse_clock(row.get("Not before")),
        }
    )

if not schedule:
    hero.warning("Add at least one match with a round to run the forecast.")
    st.stop()

try:
    forecast, finish_time, timeline_data, final_ends = schedule_forecast(
        matches,
        str(surface),
        first_start,
        int(changeover),
        schedule,
    )
except ValueError as error:
    hero.error(str(error))
    st.stop()

run_at = (
    f"{st.session_state.forecast_run_at.strftime('%H:%M')} Riga time"
)
shift_minutes = clock_minutes(shift_end)
overtime = np.maximum(0, final_ends - shift_minutes)
overtime_chance = int(round(100 * float((final_ends > shift_minutes).mean())))
overtime_likely = int(round(float(np.percentile(overtime, 50))))

with hero:
    st.markdown(
        f"**{len(schedule)} matches on a {surface.lower()} court, first one at "
        f"{first_start.strftime('%H:%M')}:**"
    )
    finish_cols = st.columns(4)
    finish_cols[0].metric("Most likely finish", format_clock(finish_time["likely"]))
    finish_cols[1].metric("Early (1 in 10 days)", format_clock(finish_time["early"]))
    finish_cols[2].metric("Late (1 in 10 days)", format_clock(finish_time["late"]))
    finish_cols[3].metric(
        f"Overtime after {shift_end.strftime('%H:%M')}",
        (f"{overtime_likely // 60} h {overtime_likely % 60:02d} min" if overtime_likely >= 60
         else f"{overtime_likely} min") if overtime_likely else "none likely",
        f"{overtime_chance}% chance of overtime",
        delta_color="off",
        delta_arrow="off",
    )
    st.caption(
        f"Based on {SIMULATIONS:,} simulated court days, each built from real match lengths. "
        f"Updated {run_at}. Change the plan below."
    )

with details_box:
    st.subheader("Match by match")
    st.altair_chart(build_timeline(timeline_data), width="stretch")
    st.caption(
        "Light bar: from the earliest likely start to the latest likely finish (1 day in 10 earlier or later). "
        "Dark bar: the most likely start and finish."
    )
    with st.expander("All numbers per match"):
        st.dataframe(forecast, hide_index=True, width="stretch")

with audit_box.expander("How long Challenger matches take, by round"):
    st.dataframe(
        historical_duration_table(matches, str(surface)),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Estimates use same-surface-and-round samples when at least "
        f"{MIN_GROUP_SIZE} matches are available; otherwise they fall back "
        "to all usable matches on the selected surface."
    )
