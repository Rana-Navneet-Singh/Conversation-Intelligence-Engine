"""Streamlit interface for the Conversation Intelligence Engine."""

import altair as alt
import pandas as pd
import streamlit as st

from chat_parser import parse_chat
from decisions import extract_decisions
from overview import get_overview
from sentiment import load_model, predict
from summary import summarize
from topics import find_topics

TEAL = "#6F9A89"
PURPLE = "#8D82B5"
TEXT = "#26332E"
MUTED = "#748078"
GRID = "#E7ECE8"

st.set_page_config(page_title="Conversation Intelligence", page_icon="💬", layout="wide")
st.markdown("""
<style>
:root { color-scheme: light; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background: #f5f6f3;
  color: #26332e;
}
[data-testid="stMainBlockContainer"] {
  max-width: 1280px;
  padding: 1.45rem 2rem 3rem;
}
[data-testid="stSidebar"] {
  background: #252c30;
  border-right: 1px solid #333b3f;
  min-width: 248px;
  max-width: 248px;
}
[data-testid="stSidebar"] > div:first-child { background: #252c30; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c2cbc7; }
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 5px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: 10px 12px;
  border: 1px solid transparent;
  border-radius: 8px;
  color: #c7cfcb;
  transition: background 120ms ease, color 120ms ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: #30393d;
  color: #ffffff;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: #38554d;
  border-color: #49675e;
  color: #ffffff;
}
[data-testid="stSidebar"] [role="radiogroup"] input { accent-color: #8bb5a4; }
.brand-row { display: flex; align-items: center; gap: 11px; margin: 4px 0 28px; }
.brand-mark {
  width: 38px; height: 38px; display: grid; place-items: center;
  border: 1px solid #59776d; border-radius: 10px; background: #344b44;
  color: #d8e9e1; font-weight: 700; font-size: 13px; letter-spacing: .04em;
}
.brand-name { color: #f5f7f6; font-size: 14px; line-height: 1.2; font-weight: 650; }
.brand-caption { color: #9ea9a4; font-size: 10px; margin-top: 4px; letter-spacing: .09em; text-transform: uppercase; }
.nav-caption { color: #89958f; font-size: 10px; letter-spacing: .14em; margin: 0 0 8px 8px; }
.sidebar-foot {
  border-top: 1px solid #3a4346; padding-top: 14px; margin-top: 28px;
  color: #9ea9a4; font-size: 11px;
}
.page-heading { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
.page-eyebrow { color: #718078; font-size: 10px; letter-spacing: .13em; font-weight: 650; text-transform: uppercase; }
.page-heading h1 { color: #26332e; font-size: 27px; line-height: 1.2; margin: 5px 0 4px; letter-spacing: -.025em; }
.page-description { color: #78837d; font-size: 13px; margin: 0 0 18px; }
.local-badge {
  display: inline-block; border: 1px solid #d9e4dd; background: #edf3ef; color: #527565;
  border-radius: 7px; padding: 6px 9px; font-size: 10px; letter-spacing: .07em; white-space: nowrap;
}
[data-testid="stMetric"] {
  background: #ffffff; border: 1px solid #e0e5e1; border-radius: 10px;
  padding: 13px 15px; box-shadow: 0 2px 5px rgba(31, 47, 39, .035); min-height: 105px;
}
[data-testid="stMetricLabel"] { color: #758079; font-size: 12px; font-weight: 550; }
[data-testid="stMetricValue"] { color: #26332e; font-size: 26px; font-weight: 650; }
[data-testid="stVerticalBlockBorderWrapper"] > div {
  background: #ffffff; border: 1px solid #e0e5e1 !important; border-radius: 10px !important;
  padding: 14px 16px;
  box-shadow: 0 2px 6px rgba(31, 47, 39, .025);
}
[data-testid="stExpander"] details {
  background: #ffffff; border: 1px solid #e0e5e1; border-radius: 10px;
  margin-bottom: 8px;
}
[data-testid="stDataFrame"], [data-testid*="Chart"] {
  background: #ffffff; border: 1px solid #e5e9e6; border-radius: 9px; padding: 5px;
}
[data-testid="stFileUploader"] {
  background: #ffffff; border: 1px dashed #cbd5cf; border-radius: 10px; padding: 10px;
}
.stButton > button {
  border-radius: 8px; min-height: 40px; font-weight: 600;
  background: #557f70; border-color: #557f70; color: #ffffff;
}
.stButton > button:hover { background: #456e5e; border-color: #456e5e; color: #ffffff; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  border-color: #d8dfda; border-radius: 8px; background: #ffffff;
}
[data-testid="stCaptionContainer"] { color: #77827b; }
h2, h3 { color: #33413b; letter-spacing: -.012em; }
hr { border-color: #e3e8e4; }
.date-range {
  display: inline-flex; align-items: center; gap: 8px; margin: 8px 0 16px;
  background: #edf3ef; border: 1px solid #dbe5de; border-radius: 7px;
  color: #526c60; padding: 6px 10px; font-size: 11px;
}
.date-range strong { color: #334a3e; font-weight: 600; }
@media (max-width: 850px) {
  [data-testid="stMainBlockContainer"] { padding: 1.1rem 1rem 2rem; }
  .page-heading h1 { font-size: 23px; }
  [data-testid="stMetric"] { min-height: 90px; padding: 10px 12px; }
}
@media (max-width: 560px) {
  [data-testid="stMainBlockContainer"] { padding: .8rem .7rem 1.5rem; }
  .page-heading { align-items: flex-start; }
  .local-badge { display: none; }
}
</style>
""", unsafe_allow_html=True)

SECTIONS = {
    "⇧  Chat Upload": ("Chat Upload", "Import a text export and preview parsed messages."),
    "▦  Overview": ("Overview", "A compact view of conversation activity."),
    "◉  Sentiment Analysis": ("Sentiment Analysis", "Message tone estimated by the trained public-data model."),
    "◈  Topic Detection": ("Topic Detection", "Explore message groups and their representative text."),
    "≡  Topic Summaries": ("Topic Summaries", "Review concise, source-based summaries for each topic."),
    "✓  Decision Detection": ("Decision Detection", "Review possible decisions with the original evidence."),
}

with st.sidebar:
    st.markdown("""
    <div class="brand-row">
      <div class="brand-mark">CI</div>
      <div>
        <div class="brand-name">Conversation Intelligence</div>
        <div class="brand-caption">Analytics workspace</div>
      </div>
    </div>
    <div class="nav-caption">WORKSPACE</div>
    """, unsafe_allow_html=True)
    nav_label = st.radio("Navigation", list(SECTIONS), label_visibility="collapsed", key="main_navigation")
    st.markdown("<div class='sidebar-foot'>Local analysis · TXT conversations</div>", unsafe_allow_html=True)

section, description = SECTIONS[nav_label]
st.markdown(f"""
<div class="page-heading">
  <div><div class="page-eyebrow">Conversation intelligence</div><h1>{section}</h1></div>
  <span class="local-badge">LOCAL WORKSPACE</span>
</div>
<p class="page-description">{description}</p>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "topics" not in st.session_state:
    st.session_state.topics = []
if "sentiments" not in st.session_state:
    st.session_state.sentiments = []

if section == "Chat Upload":
    with st.container(border=True):
        st.subheader("Upload a conversation")
        st.caption("Choose a WhatsApp-exported TXT file. Media is not analyzed.")
        uploaded = st.file_uploader("WhatsApp TXT export", type=["txt"], label_visibility="collapsed")
        if st.button("Analyze chat", type="primary", disabled=uploaded is None):
            try:
                parsed = parse_chat(uploaded.getvalue().decode("utf-8-sig", errors="replace"))
                texts = [row["message"].strip() for row in parsed
                         if isinstance(row.get("message"), str) and row["message"].strip()]
                st.session_state.messages = parsed
                st.session_state.topics = []
                st.session_state.sentiments = predict(texts)
                st.session_state.parse_notice = None if parsed else (
                    "No messages were parsed. Check that the TXT file contains timestamped WhatsApp messages.")
                st.session_state.topic_notice = None
                if len(texts) < 2 or len(set(texts)) < 2:
                    st.session_state.topic_notice = (
                        "Topic detection needs at least two non-empty, different text messages.")
                elif parsed:
                    with st.spinner("Finding topics..."):
                        try:
                            st.session_state.topics = find_topics(parsed)
                            if not st.session_state.topics:
                                st.session_state.topic_notice = (
                                    "No coherent topic groups were found. Try a chat with more descriptive messages.")
                        except ImportError:
                            st.session_state.topic_notice = (
                                "Topic embeddings are not installed yet. Install this project's requirements to enable topics.")
                        except Exception as topic_error:
                            st.session_state.topic_notice = f"Topic analysis could not run: {topic_error}"
            except Exception as error:
                st.session_state.messages = []
                st.session_state.topics = []
                st.session_state.sentiments = []
                st.error(f"Could not analyze this file: {error}")

    messages = st.session_state.messages
    if not messages:
        st.info(st.session_state.get("parse_notice") or
                "Upload a text export and select Analyze chat to populate the dashboard.")
    else:
        with st.container(border=True):
            st.subheader("Parsed message preview")
            st.dataframe(messages[:100], use_container_width=True, hide_index=True)
            st.caption(f"Showing up to 100 of {len(messages)} parsed messages.")

else:
    messages = st.session_state.messages
    if not messages:
        with st.container(border=True):
            st.info("Upload and analyze a WhatsApp TXT export to see this section.")
    elif section == "Overview":
        info = get_overview(messages)
        st.markdown(f"<div class='date-range'>CONVERSATION PERIOD <strong>{info['start'].date()} – {info['end'].date()}</strong></div>",
                    unsafe_allow_html=True)
        metric_columns = st.columns(3)
        metric_columns[0].metric("Total messages", info["total_messages"])
        metric_columns[1].metric("Participants", info["participants"])
        metric_columns[2].metric("Most active", info["most_active"] or "—")

        activity_column, people_column = st.columns([1.35, 1], gap="large")
        with activity_column:
            with st.container(border=True):
                st.subheader("Conversation activity")
                st.caption("Messages over time")
                activity_data = pd.DataFrame(info["activity"], columns=["Date", "Messages"])
                activity_data["Date"] = pd.to_datetime(activity_data["Date"])
                activity_chart = alt.Chart(activity_data).mark_line(
                    color=PURPLE, strokeWidth=2.5, interpolate="monotone",
                    point=alt.OverlayMarkDef(color=PURPLE, filled=True, size=34)
                ).encode(
                    x=alt.X("Date:T", title=None,
                            axis=alt.Axis(format="%b %d", tickCount=6, grid=False)),
                    y=alt.Y("Messages:Q", title="Messages", scale=alt.Scale(zero=True),
                            axis=alt.Axis(tickMinStep=1, grid=True, gridColor=GRID)),
                    tooltip=[alt.Tooltip("Date:T", title="Date"),
                             alt.Tooltip("Messages:Q", title="Messages")],
                ).properties(height=240)
                st.altair_chart(activity_chart, use_container_width=True)
        with people_column:
            with st.container(border=True):
                st.subheader("Active participants")
                st.caption("Ranked by message count")
                people_data = pd.DataFrame(info["active_participants"][:8],
                                           columns=["Participant", "Messages"])
                people_chart = alt.Chart(people_data).mark_bar(
                    color=TEAL, cornerRadiusEnd=4, size=17
                ).encode(
                    x=alt.X("Messages:Q", title="Messages", scale=alt.Scale(zero=True),
                            axis=alt.Axis(tickMinStep=1, grid=True, gridColor=GRID)),
                    y=alt.Y("Participant:N", title=None, sort="-x",
                            axis=alt.Axis(labelLimit=140, grid=False)),
                    tooltip=[alt.Tooltip("Participant:N"), alt.Tooltip("Messages:Q")],
                ).properties(height=240)
                st.altair_chart(people_chart, use_container_width=True)

    elif section == "Sentiment Analysis":
        labels = st.session_state.sentiments
        model = load_model()
        if model is None:
            with st.container(border=True):
                st.warning("The sentiment model is not available. Train it with the included public-data workflow first.")
        elif not labels:
            with st.container(border=True):
                st.info("There are no non-empty messages to classify.")
        else:
            sentiment_order = ["positive", "neutral", "negative"]
            counts = {label: labels.count(label) for label in sentiment_order}
            positive_share = counts["positive"] / len(labels)
            metric_columns = st.columns(3)
            metric_columns[0].metric("Positive messages", counts["positive"])
            metric_columns[1].metric("Neutral messages", counts["neutral"])
            metric_columns[2].metric("Negative messages", counts["negative"])
            chart_column, table_column = st.columns([1, 1.2], gap="large")
            with chart_column:
                with st.container(border=True):
                    st.subheader("Overall distribution")
                    sentiment_data = pd.DataFrame(
                        [{"Sentiment": name.title(), "Messages": count} for name, count in counts.items()])
                    sentiment_chart = alt.Chart(sentiment_data).mark_bar(
                        cornerRadiusEnd=5, size=28
                    ).encode(
                        x=alt.X("Messages:Q", title="Messages", scale=alt.Scale(zero=True),
                                axis=alt.Axis(tickMinStep=1, grid=True, gridColor=GRID)),
                        y=alt.Y("Sentiment:N", title=None,
                                sort=["Positive", "Neutral", "Negative"], axis=alt.Axis(grid=False)),
                        color=alt.Color("Sentiment:N", legend=None,
                                        scale=alt.Scale(domain=["Positive", "Neutral", "Negative"],
                                                        range=[TEAL, PURPLE, "#C78F8E"])),
                        tooltip=[alt.Tooltip("Sentiment:N"), alt.Tooltip("Messages:Q")],
                    ).properties(height=190)
                    st.altair_chart(sentiment_chart, use_container_width=True)
                    st.caption(f"Positive share: {positive_share:.0%} · {len(labels)} messages scored")
            with table_column:
                with st.container(border=True):
                    st.subheader("By participant")
                    participant_counts = []
                    for sender in sorted({row["sender"] for row in messages}):
                        selected = [labels[i] for i, row in enumerate(messages) if row["sender"] == sender]
                        participant_counts.append({"Participant": sender,
                                                   **{name.title(): selected.count(name)
                                                      for name in sentiment_order}})
                    st.dataframe(participant_counts, use_container_width=True, hide_index=True)
                    st.caption("English TweetEval training data may not reflect Hinglish, slang, or sarcasm.")

    elif section == "Topic Detection":
        topics = st.session_state.topics
        if not topics:
            with st.container(border=True):
                st.info(st.session_state.get("topic_notice") or
                        "No topic groups were found. Check that the chat has enough distinct text messages.")
        else:
            st.caption(f"{len(topics)} groups · Representative messages are closest to each group center.")
            for topic in topics:
                with st.expander(f"{topic['name']}   ·   {len(topic['messages'])} messages · {len(topic.get('participants', []))} participants"):
                    st.caption("Keywords: " + (", ".join(topic.get("keywords", [])) or "not available"))
                    st.caption("Semantic multilingual embeddings · CPU · variable topic count. Review examples for quality.")
                    st.write("Participants: " + (", ".join(topic.get("participants", [])) or "Unknown"))
                    for row in topic["representative_messages"]:
                        st.markdown(f"**{row['sender']}**  ·  {row['timestamp']}")
                        st.write(row["message"])

    elif section == "Topic Summaries":
        topics = st.session_state.topics
        if not topics:
            with st.container(border=True):
                st.info("Analyze a chat with enough distinct text to create topic summaries.")
        else:
            selected = st.selectbox("Choose a topic", topics, format_func=lambda topic: topic["name"])
            result = summarize(selected["messages"], topic_title=selected["name"])
            with st.container(border=True):
                st.subheader(result["title"])
                st.caption("Chronological source-based summary · original wording retained")
                for entry in result["timeline"]:
                    stamp = entry["timestamp"]
                    when = stamp.strftime("%b %d, %H:%M") if hasattr(stamp, "strftime") else "Time unknown"
                    st.markdown(f"**{entry['sender']}** · {when} · _{entry['role']}_")
                    st.write(entry["text"])
                st.markdown("**Participants**")
                st.write(", ".join(result["participants"]) or "Unknown")
                if result["dates"]:
                    st.markdown("**Dates**")
                    st.write("  ·  ".join(result["dates"]))
                st.markdown("**Conclusion**")
                st.write(result["conclusion"])
                st.markdown("**Tasks and commitments**")
                if result["tasks"]:
                    for task in result["tasks"]:
                        deadline = f" · Deadline: {task['deadline']}" if task["deadline"] else ""
                        st.write(f"{task['responsible']} — {task['task']}{deadline}")
                else:
                    st.caption("No explicit commitment identified.")
            with st.container(border=True):
                st.subheader("Supporting messages")
                st.dataframe(selected["messages"], use_container_width=True, hide_index=True)

    elif section == "Decision Detection":
        decisions = extract_decisions(messages)
        if not decisions:
            with st.container(border=True):
                st.info("No clear decision phrases were found. This does not prove that no decision was made.")
        for decision in decisions:
            with st.container(border=True):
                confidence = decision.get("confidence", "uncertain")
                st.markdown(f"**{decision['status'].title()} · {decision['type'].title()}**  ·  {confidence} confidence")
                st.write(decision["decision"])
                st.caption(f"Proposer: {decision['proposer']} · Agreed by: "
                           + (", ".join(decision["agreed_by"]) or "not identified")
                           + f" · Responsible: {decision['responsible'] or 'not identified'}"
                           + f" · Explicit deadline: {decision['deadline'] or 'not stated'}")
                with st.expander("Evidence messages"):
                    for evidence in decision["evidence"]:
                        st.markdown(f"**{evidence['role'].title()}** · {evidence.get('sender', 'Unknown')} · {evidence.get('timestamp', 'Time unknown')}")
                        st.write(evidence["message"])
