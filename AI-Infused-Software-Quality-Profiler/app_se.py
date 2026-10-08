"""AI-Infused Software Quality Profiler.

Automated Static Metrics Engine & Structural Code Analysis Architecture.
Lead Developer & Researcher: Prajnaa M.

Run with:  streamlit run app_se.py
"""

import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "AI-Infused Software Quality Profiler"
APP_SUBTITLE = "Automated Static Metrics Engine & Structural Code Analysis Architecture"
AUTHOR_CREDIT = "Lead Developer & Researcher: Prajnaa M."

NAV_INGESTION = "📊 Code Metrics Ingestion"
NAV_INSIGHTS = "🔬 Statistical Insights & Context"

DECISION_KEYWORDS = ("if", "for", "while", "elif", "def", "except")
NESTING_THRESHOLD = 3
INDENT_WIDTH = 4

DECISION_PATTERN = re.compile(r"\b(" + "|".join(DECISION_KEYWORDS) + r")\b")
CONTROL_PATTERN = re.compile(r"^\s*(if|elif|else|for|while|try|except|finally|with)\b")
LOOP_PATTERN = re.compile(r"^\s*(for|while)\b")
TRY_PATTERN = re.compile(r"^\s*try\s*:")
BARE_EXCEPT_PATTERN = re.compile(r"^\s*except\s*:")
FUNCTION_PATTERN = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)")
IO_PATTERN = re.compile(
    r"\b(open|urlopen|read_csv|read_json|read_excel|to_csv|connect)\s*\(|"
    r"\brequests\.(get|post|put|delete)\s*\("
)
NON_CONTEXT_OPEN_PATTERN = re.compile(r"=\s*open\s*\(")
COMPREHENSION_PATTERN = re.compile(r"\[[^\]]*\bfor\b[^\]]*\bin\b[^\]]*\]")
NESTED_COMPREHENSION_PATTERN = re.compile(r"\[[^\]]*\bfor\b[^\]]*\bfor\b[^\]]*\]")
APPEND_PATTERN = re.compile(r"\.append\s*\(")
LOOKUP_PATTERN = re.compile(r"""\b\w+\[(?:"[^"]+"|'[^']+')\]""")
STRING_PATTERN = re.compile(r"""("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""")

DEFAULT_CODE = '''import csv
import statistics

# Load transaction records from disk
def load_records(path):
    """Read a CSV file and return a list of row dictionaries."""
    records = []
    f = open(path, "r")
    reader = csv.DictReader(f)
    for row in reader:
        records.append(row)
    f.close()
    return records

# Clean and aggregate values by category
def summarize(records, threshold):
    totals = {}
    for row in records:
        category = row["category"]
        amount = float(row["amount"])
        if amount > threshold:
            if category in totals:
                totals[category].append(amount)
            else:
                totals[category] = [amount]
        elif amount < 0:
            continue
    report = {}
    for category in totals:
        values = totals[category]
        report[category] = {"mean": statistics.mean(values), "count": len(values)}
    return report

def main():
    records = load_records("transactions.csv")
    attempts = 0
    while attempts < 3:
        try:
            report = summarize(records, 100.0)
            break
        except KeyError:
            attempts += 1
    for name in report:
        print(name, report[name]["mean"])

main()
'''

APP_STYLES = """
<style>
.hero-banner {background: linear-gradient(120deg, #1e3a8a 0%, #6d28d9 100%); padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1.2rem; box-shadow: 0 6px 18px rgba(30, 58, 138, 0.25);}
.hero-title {font-size: 2.1rem; font-weight: 800; color: #ffffff; line-height: 1.2; margin: 0;}
.hero-subtitle {font-size: 1.02rem; color: #e0e7ff; margin-top: 0.45rem;}
.credit-card {background: linear-gradient(135deg, rgba(109, 40, 217, 0.18), rgba(30, 58, 138, 0.18)); border: 1px solid rgba(109, 40, 217, 0.45); border-radius: 12px; padding: 0.9rem 1rem; margin-bottom: 0.8rem;}
.credit-label {font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.75;}
.credit-name {font-size: 1.05rem; font-weight: 700; margin-top: 0.2rem;}
.credit-venue {font-size: 0.8rem; opacity: 0.8; margin-top: 0.2rem;}
</style>
"""

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceLine:
    """A single classified line of the submitted source code."""

    number: int
    raw: str
    kind: str
    indent: int
    code: str


@dataclass(frozen=True)
class FunctionProfile:
    """Per-function static metrics."""

    name: str
    start_line: int
    length: int
    complexity: int
    nesting: int
    has_docstring: bool


@dataclass(frozen=True)
class CodeProfile:
    """Aggregate static metrics for a source file."""

    total_lines: int
    blank_lines: int
    code_lines: int
    comment_lines: int
    complexity: int
    comment_density: float
    max_nesting: int
    keyword_counts: Dict[str, int]
    functions: List[FunctionProfile]
    deep_lines: List[int]
    unguarded_io: List[int]
    non_context_open: List[int]
    bare_except: List[int]
    comprehension_hotspots: List[int]
    loop_appends: List[int]
    repeated_lookups: Dict[str, int]
    has_main_guard: bool


@dataclass(frozen=True)
class ScoreBreakdown:
    """Maintainability index with itemised penalty points."""

    complexity_penalty: float
    documentation_penalty: float
    nesting_penalty: float
    resilience_penalty: float
    score: float
    rating: str
    badge: str


# ---------------------------------------------------------------------------
# Static analysis engine
# ---------------------------------------------------------------------------


def sanitize_code(line: str) -> str:
    """Remove string literals and trailing comments so keywords inside them are ignored."""
    without_strings = STRING_PATTERN.sub('""', line)
    return without_strings.split("#", 1)[0]


def classify_source(source: str) -> List[SourceLine]:
    """Classify every line as blank, comment, docstring, or code."""
    classified: List[SourceLine] = []
    inside_docstring = False
    delimiter = ""
    for number, raw in enumerate(source.splitlines(), start=1):
        expanded = raw.expandtabs(INDENT_WIDTH)
        stripped = expanded.strip()
        indent = len(expanded) - len(expanded.lstrip())
        if inside_docstring:
            classified.append(SourceLine(number, raw, "docstring", indent, ""))
            if delimiter in stripped:
                inside_docstring = False
            continue
        if not stripped:
            classified.append(SourceLine(number, raw, "blank", indent, ""))
        elif stripped.startswith("#"):
            classified.append(SourceLine(number, raw, "comment", indent, ""))
        elif stripped.startswith('"""') or stripped.startswith("'''"):
            delimiter = stripped[:3]
            if delimiter not in stripped[3:]:
                inside_docstring = True
            classified.append(SourceLine(number, raw, "docstring", indent, ""))
        else:
            classified.append(SourceLine(number, raw, "code", indent, sanitize_code(expanded)))
    return classified


def collect_ancestors(lines: List[SourceLine], index: int) -> List[SourceLine]:
    """Return the chain of enclosing code lines using indentation."""
    ancestors: List[SourceLine] = []
    reference = lines[index].indent
    for cursor in range(index - 1, -1, -1):
        candidate = lines[cursor]
        if candidate.kind != "code":
            continue
        if candidate.indent < reference:
            ancestors.append(candidate)
            reference = candidate.indent
            if reference == 0:
                break
    return ancestors


def profile_functions(lines: List[SourceLine], depth_by_line: Dict[int, int]) -> List[FunctionProfile]:
    """Compute complexity, length, nesting, and docstring presence for each function."""
    profiles: List[FunctionProfile] = []
    for index, line in enumerate(lines):
        if line.kind != "code":
            continue
        match = FUNCTION_PATTERN.match(line.code)
        if not match:
            continue
        body: List[SourceLine] = []
        for follower in lines[index + 1:]:
            if follower.kind == "code" and follower.indent <= line.indent:
                break
            body.append(follower)
        while body and body[-1].kind in ("blank", "comment"):
            body.pop()
        code_body = [item for item in body if item.kind == "code"]
        decisions = sum(len(DECISION_PATTERN.findall(item.code)) for item in code_body)
        nesting = max((depth_by_line.get(item.number, 0) for item in code_body), default=0)
        first_meaningful = next((item for item in body if item.kind not in ("blank", "comment")), None)
        has_docstring = first_meaningful is not None and first_meaningful.kind == "docstring"
        profiles.append(
            FunctionProfile(
                name=match.group(1),
                start_line=line.number,
                length=1 + len(body),
                complexity=1 + decisions,
                nesting=nesting,
                has_docstring=has_docstring,
            )
        )
    return profiles


def analyze_source(source: str) -> CodeProfile:
    """Run the full deterministic analysis pipeline on the submitted source."""
    lines = classify_source(source)
    total_lines = len(lines)
    blank_lines = sum(1 for item in lines if item.kind == "blank")
    code_lines = sum(1 for item in lines if item.kind == "code")
    comment_lines = sum(1 for item in lines if item.kind in ("comment", "docstring"))

    keyword_tally: Counter = Counter()
    for item in lines:
        if item.kind == "code":
            keyword_tally.update(DECISION_PATTERN.findall(item.code))
    keyword_counts = {keyword: keyword_tally.get(keyword, 0) for keyword in DECISION_KEYWORDS}
    complexity = 1 + sum(keyword_counts.values())
    comment_density = (comment_lines / total_lines * 100.0) if total_lines else 0.0

    depth_by_line: Dict[int, int] = {}
    deep_lines: List[int] = []
    unguarded_io: List[int] = []
    non_context_open: List[int] = []
    bare_except: List[int] = []
    comprehension_hotspots: List[int] = []
    loop_appends: List[int] = []
    lookup_tally: Counter = Counter()

    for index, item in enumerate(lines):
        if item.kind != "code":
            continue
        ancestors = collect_ancestors(lines, index)
        in_loop = any(LOOP_PATTERN.match(ancestor.code) for ancestor in ancestors)

        if CONTROL_PATTERN.match(item.code):
            depth = 1 + sum(1 for ancestor in ancestors if CONTROL_PATTERN.match(ancestor.code))
            depth_by_line[item.number] = depth
            if depth >= NESTING_THRESHOLD:
                deep_lines.append(item.number)

        if IO_PATTERN.search(item.code):
            if not any(TRY_PATTERN.match(ancestor.code) for ancestor in ancestors):
                unguarded_io.append(item.number)
        if NON_CONTEXT_OPEN_PATTERN.search(item.code):
            non_context_open.append(item.number)
        if BARE_EXCEPT_PATTERN.match(item.code):
            bare_except.append(item.number)
        if COMPREHENSION_PATTERN.search(item.code) and (in_loop or NESTED_COMPREHENSION_PATTERN.search(item.code)):
            comprehension_hotspots.append(item.number)
        if in_loop and APPEND_PATTERN.search(item.code):
            loop_appends.append(item.number)
        lookup_tally.update(LOOKUP_PATTERN.findall(item.raw.split("#", 1)[0]))

    functions = profile_functions(lines, depth_by_line)
    repeated_lookups = {expr: count for expr, count in lookup_tally.items() if count >= 3}
    has_main_guard = "__name__" in source

    return CodeProfile(
        total_lines=total_lines,
        blank_lines=blank_lines,
        code_lines=code_lines,
        comment_lines=comment_lines,
        complexity=complexity,
        comment_density=comment_density,
        max_nesting=max(depth_by_line.values(), default=0),
        keyword_counts=keyword_counts,
        functions=functions,
        deep_lines=deep_lines,
        unguarded_io=unguarded_io,
        non_context_open=non_context_open,
        bare_except=bare_except,
        comprehension_hotspots=comprehension_hotspots,
        loop_appends=loop_appends,
        repeated_lookups=repeated_lookups,
        has_main_guard=has_main_guard,
    )


def rate_score(score: float) -> Tuple[str, str]:
    """Translate a numeric maintainability score into a rating and badge."""
    if score >= 85:
        return "Highly Maintainable", "🟢"
    if score >= 65:
        return "Moderately Maintainable", "🟡"
    if score >= 40:
        return "Difficult to Maintain", "🟠"
    return "Critical Maintenance Risk", "🔴"


def compute_maintainability(profile: CodeProfile) -> ScoreBreakdown:
    """Score the code profile from 0 to 100 using complexity-to-comment ratios and risk penalties."""
    complexity_density = profile.complexity / max(profile.code_lines, 1) * 100.0
    complexity_penalty = min(45.0, complexity_density * 0.9)
    documentation_penalty = min(16.0, max(0.0, 20.0 - profile.comment_density) * 0.8)
    nesting_penalty = min(18.0, max(0, profile.max_nesting - 2) * 6.0)
    resilience_penalty = min(
        21.0,
        5.0 * len(profile.unguarded_io) + 3.0 * len(profile.bare_except) + 2.0 * len(profile.non_context_open),
    )
    raw_score = 100.0 - complexity_penalty - documentation_penalty - nesting_penalty - resilience_penalty
    score = round(max(0.0, min(100.0, raw_score)), 1)
    rating, badge = rate_score(score)
    return ScoreBreakdown(
        complexity_penalty=round(complexity_penalty, 1),
        documentation_penalty=round(documentation_penalty, 1),
        nesting_penalty=round(nesting_penalty, 1),
        resilience_penalty=round(resilience_penalty, 1),
        score=score,
        rating=rating,
        badge=badge,
    )


def classify_complexity(value: int) -> Tuple[str, str]:
    """Map a cyclomatic complexity estimate to a risk band."""
    if value <= 10:
        return "Low risk", "Simple structure that is straightforward to test."
    if value <= 20:
        return "Moderate risk", "Moderately complex; review for extraction opportunities."
    if value <= 50:
        return "High risk", "Complex logic; testing effort grows quickly."
    return "Very high risk", "Practically untestable without restructuring."


def function_risk(complexity: int) -> str:
    """Per-function risk label used in tables and findings."""
    if complexity <= 5:
        return "Low"
    if complexity <= 10:
        return "Moderate"
    return "High"


# ---------------------------------------------------------------------------
# Review generation (heuristic simulation of an LLM architecture review)
# ---------------------------------------------------------------------------


def format_line_refs(numbers: List[int], limit: int = 6) -> str:
    """Format line numbers as a compact reference string."""
    shown = ", ".join(f"L{number}" for number in numbers[:limit])
    remaining = len(numbers) - limit
    if remaining > 0:
        return f"{shown} and {remaining} more"
    return shown


def build_critical_findings(profile: CodeProfile) -> List[str]:
    """Create the list of critical structural findings."""
    findings: List[str] = []
    if profile.max_nesting >= NESTING_THRESHOLD:
        findings.append(
            f"**Deeply nested control blocks detected**: maximum nesting depth is {profile.max_nesting} "
            f"(threshold {NESTING_THRESHOLD}), first reached at {format_line_refs(profile.deep_lines)}. "
            "Every extra level multiplies the execution paths that must be tested and hides unreachable branches."
        )
    if profile.unguarded_io:
        findings.append(
            f"**Missing try-except on I/O transactions** at {format_line_refs(profile.unguarded_io)}: "
            "file, network, or database calls are not wrapped in exception handling, so a missing file, "
            "permission error, or encoding failure will surface as an unhandled crash."
        )
    if profile.non_context_open:
        findings.append(
            f"**Manually managed resource handle** at {format_line_refs(profile.non_context_open)}: "
            "`open()` is assigned without a context manager, so the handle leaks whenever an exception "
            "occurs before `close()` is reached."
        )
    if profile.bare_except:
        findings.append(
            f"**Bare `except:` clause** at {format_line_refs(profile.bare_except)}: it swallows "
            "`KeyboardInterrupt` and `SystemExit` and conceals the real failure cause."
        )
    if profile.comprehension_hotspots:
        findings.append(
            f"**Unoptimized list comprehension overhead** at {format_line_refs(profile.comprehension_hotspots)}: "
            "comprehensions are nested or re-built inside loops, repeatedly allocating intermediate lists. "
            "Prefer generator expressions or hoist the construction out of the loop."
        )
    if not findings:
        findings.append(
            "**No critical structural vulnerabilities detected** by the heuristic pass: nesting depth, "
            "I/O guarding, and resource handling all fall within acceptable thresholds."
        )
    return findings


def build_recommendations(profile: CodeProfile) -> List[str]:
    """Create the list of refactoring and maintainability recommendations."""
    recommendations: List[str] = []
    heavy_functions = [
        func for func in profile.functions if func.complexity >= 6 or func.nesting >= NESTING_THRESHOLD
    ]
    for func in heavy_functions:
        recommendations.append(
            f"**High cognitive complexity in `{func.name}()`** (line {func.start_line}): cyclomatic estimate "
            f"{func.complexity}, nesting depth {func.nesting}, {func.length} lines. Extract the inner "
            "branches into small, single-purpose helper methods with descriptive names."
        )
    if profile.loop_appends:
        recommendations.append(
            f"**Repetitive accumulation pattern** at {format_line_refs(profile.loop_appends)}: `.append()` "
            "inside loops can often be replaced with comprehensions, `collections.defaultdict(list)`, or "
            "`dict.setdefault()` to remove the membership-check branches."
        )
    for expression, count in sorted(profile.repeated_lookups.items(), key=lambda pair: -pair[1]):
        recommendations.append(
            f"**Repeated lookup `{expression}`** appears {count} times: bind it to a local variable "
            "or a localized dictionary once, then reuse it."
        )
    undocumented = [func.name for func in profile.functions if not func.has_docstring]
    if undocumented:
        names = ", ".join(f"`{name}()`" for name in undocumented[:6])
        recommendations.append(
            f"**Missing docstrings** for {names}: document parameters, return values, and raised exceptions."
        )
    if profile.comment_density < 15.0:
        recommendations.append(
            f"**Low documentation density** ({profile.comment_density:.1f}% versus a 15-20% target): add "
            "intent-level comments explaining why decisions are made, not what the syntax does."
        )
    if profile.functions and not profile.has_main_guard:
        recommendations.append(
            "**No entry-point guard**: wrap top-level execution in `if __name__ == \"__main__\":` so the "
            "module can be imported and unit-tested without side effects."
        )
    if not recommendations:
        recommendations.append(
            "**No refactoring blockers identified**: function sizes, documentation, and branching are healthy. "
            "Maintain this profile by enforcing complexity thresholds in continuous integration."
        )
    return recommendations


def render_bullets(items: List[str]) -> None:
    """Render a list of markdown strings as bullets."""
    st.markdown("\n".join(f"- {item}" for item in items))


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------


def initialize_state() -> None:
    """Create session-state defaults exactly once."""
    if "source_code" not in st.session_state:
        st.session_state.source_code = DEFAULT_CODE
    if "reviewed_source" not in st.session_state:
        st.session_state.reviewed_source = None


def inject_styles() -> None:
    """Inject the custom stylesheet."""
    st.markdown(APP_STYLES, unsafe_allow_html=True)


def render_sidebar() -> str:
    """Render the sidebar and return the selected page."""
    with st.sidebar:
        st.markdown("### 🧬 Quality Profiler")
        st.markdown(
            '<div class="credit-card">'
            '<div class="credit-label">Author Credit</div>'
            f'<div class="credit-name">{AUTHOR_CREDIT}</div>'
            '<div class="credit-venue">ACM Winter School · IIIT Bangalore</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", [NAV_INGESTION, NAV_INSIGHTS], label_visibility="collapsed")
        st.divider()
        st.markdown("**System Technical Highlights**")
        st.markdown("- 🐍 Python 3.10+\n- 🧱 Streamlit Native Layout\n- 🔎 Regex Parsing")
    return page


def render_header() -> None:
    """Render the hero banner."""
    st.markdown(
        '<div class="hero-banner">'
        f'<div class="hero-title">{APP_TITLE}</div>'
        f'<div class="hero-subtitle">{APP_SUBTITLE}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the page footer."""
    st.divider()
    st.caption(f"{AUTHOR_CREDIT} · ACM Winter School Portfolio · IIIT Bangalore")


def render_metric_columns(profile: CodeProfile) -> None:
    """Render the three headline metrics."""
    band, band_text = classify_complexity(profile.complexity)
    col_loc, col_cc, col_comments = st.columns(3)
    with col_loc:
        st.metric(
            "Total Lines of Code (LOC)",
            profile.total_lines,
            help="All physical lines submitted, including blank and comment lines.",
        )
        st.caption(f"{profile.code_lines} executable · {profile.blank_lines} blank · {profile.comment_lines} documentation")
    with col_cc:
        st.metric(
            "Cyclomatic Complexity Estimate",
            profile.complexity,
            help="1 + occurrences of if, for, while, elif, def, and except in executable code.",
        )
        st.caption(f"{band}: {band_text}")
    with col_comments:
        st.metric(
            "Comments Density Ratio",
            f"{profile.comment_density:.1f}%",
            help="Lines starting with # or triple quotes (including docstring bodies) divided by total lines.",
        )
        st.caption("Recommended range: 15% to 20%")


def render_review(profile: CodeProfile) -> None:
    """Render the three review expanders."""
    breakdown = compute_maintainability(profile)

    with st.expander("🔴 Critical Structural Vulnerabilities", expanded=True):
        render_bullets(build_critical_findings(profile))

    with st.expander("🟡 Refactoring & Maintainability Recommendations", expanded=True):
        render_bullets(build_recommendations(profile))

    with st.expander("🟢 Automated Maintainability Index Score", expanded=True):
        score_col, detail_col = st.columns([1, 2])
        with score_col:
            st.metric("Maintainability Index", f"{breakdown.score:.1f} / 100")
            st.markdown(f"**{breakdown.badge} {breakdown.rating}**")
        with detail_col:
            st.progress(breakdown.score / 100.0, text=f"Profile rating: {breakdown.score:.1f} out of 100")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Scoring Factor": [
                            "Complexity-to-code density",
                            "Comment and docstring deficit",
                            "Control-flow nesting depth",
                            "Error-handling and resource risk",
                        ],
                        "Penalty Points": [
                            breakdown.complexity_penalty,
                            breakdown.documentation_penalty,
                            breakdown.nesting_penalty,
                            breakdown.resilience_penalty,
                        ],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )


def render_ingestion_page() -> None:
    """Page 1: code ingestion, deterministic metrics, and the simulated review."""
    st.subheader("📊 Code Metrics Ingestion")
    source = st.text_area(
        "Paste Python source code for analysis",
        value=st.session_state.source_code,
        height=430,
        key="code_editor",
    )
    st.session_state.source_code = source

    if not source.strip():
        st.warning("Paste some Python code above to compute metrics.")
        return

    profile = analyze_source(source)
    st.markdown("#### Deterministic Metrics")
    render_metric_columns(profile)

    st.divider()
    st.markdown("#### Structural Architecture Review")
    st.caption(


        "The review is a simulated LLM experience: findings are generated from the deterministic "
        "static-analysis results above, and no external model is called."
    )

    if st.button("🚀 Execute Structural Architecture Review", type="primary"):
        with st.spinner("Parsing AST nodes and executing code-path heuristics..."):
            time.sleep(1.6)
            analyze_source(source)
            time.sleep(1.4)
        st.session_state.reviewed_source = source
        st.success("Architecture review complete.")

    if st.session_state.reviewed_source == source:
        render_review(profile)
    elif st.session_state.reviewed_source is not None:
        st.info("The code changed since the last review. Run the review again to refresh the findings.")


def render_insights_page() -> None:
    """Page 2: statistical context for the current code profile."""
    st.subheader("🔬 Statistical Insights & Context")
    source = st.session_state.source_code
    if not source.strip():
        st.warning("No code available. Open Code Metrics Ingestion and paste a Python script first.")
        return

    profile = analyze_source(source)
    breakdown = compute_maintainability(profile)
    band, band_text = classify_complexity(profile.complexity)

    st.info(
        f"Current profile: cyclomatic estimate **{profile.complexity}** ({band}), maintainability index "
        f"**{breakdown.score:.1f}** ({breakdown.badge} {breakdown.rating}). {band_text}"
    )

    tab_keywords, tab_functions, tab_method, tab_limits = st.tabs(
        ["Keyword Distribution", "Function Profiles", "Methodology", "Limitations"]
    )

    with tab_keywords:
        st.markdown("Decision-keyword frequency driving the cyclomatic complexity estimate.")
        keyword_frame = pd.DataFrame(
            {"Occurrences": [profile.keyword_counts[keyword] for keyword in DECISION_KEYWORDS]},
            index=list(DECISION_KEYWORDS),
        )
        st.bar_chart(keyword_frame)
        st.markdown("Penalty points deducted from a perfect score of 100.")
        penalty_frame = pd.DataFrame(
            {
                "Penalty Points": [
                    breakdown.complexity_penalty,
                    breakdown.documentation_penalty,
                    breakdown.nesting_penalty,
                    breakdown.resilience_penalty,
                ]
            },
            index=["Complexity", "Documentation", "Nesting", "Resilience"],
        )
        st.bar_chart(penalty_frame)

    with tab_functions:
        if profile.functions:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Function": [func.name for func in profile.functions],
                        "Start Line": [func.start_line for func in profile.functions],
                        "Length (lines)": [func.length for func in profile.functions],
                        "Complexity": [func.complexity for func in profile.functions],
                        "Max Nesting": [func.nesting for func in profile.functions],
                        "Docstring": ["Yes" if func.has_docstring else "No" for func in profile.functions],
                        "Risk": [function_risk(func.complexity) for func in profile.functions],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No function definitions were found in the submitted code.")
        st.markdown("**Cyclomatic complexity reference bands**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Complexity": ["1 - 10", "11 - 20", "21 - 50", "Above 50"],
                    "Risk Band": ["Low risk", "Moderate risk", "High risk", "Very high risk"],
                    "Interpretation": [
                        "Simple structure that is straightforward to test.",
                        "Moderately complex; review for extraction opportunities.",
                        "Complex logic; testing effort grows quickly.",
                        "Practically untestable without restructuring.",
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with tab_method:
        st.markdown(
            "**Cyclomatic complexity estimate**: `CC = 1 + count(if, for, while, elif, def, except)`, "
            "evaluated on executable lines after string literals and comments are stripped. The regex "
            "uses word boundaries, so `elif` is never double-counted as `if`."
        )
        st.markdown(
            "**Comments density**: `(comment lines + docstring lines) / total lines x 100`. A line counts "
            "when it starts with `#` or a triple quote, and docstring body lines are included."
        )
        st.markdown(
            "**Maintainability index**: `100 - complexity penalty - documentation penalty - nesting penalty "
            "- resilience penalty`, clamped to the range 0 to 100."
        )
        st.markdown(
            "- Complexity penalty: `min(45, 0.9 x complexity per 100 code lines)`\n"
            "- Documentation penalty: `min(16, 0.8 x max(0, 20 - comment density))`\n"
            "- Nesting penalty: `min(18, 6 x max(0, depth - 2))`\n"
            "- Resilience penalty: `5` per unguarded I/O call, `3` per bare except, `2` per unmanaged `open()`, capped at 21"
        )

    with tab_limits:
        st.markdown(
            "- Analysis is lexical and indentation-based; it does not build a full abstract syntax tree.\n"
            "- Multi-line expressions with unusual indentation can distort nesting-depth estimates.\n"
            "- Keywords used in comprehensions and conditional expressions are counted as decision points.\n"
            "- The architecture review is a heuristic simulation, not a call to a language model.\n"
            "- Scores are comparative indicators and should complement, not replace, code review."
        )


def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_state()
    inject_styles()
    page = render_sidebar()
    render_header()
    if page == NAV_INGESTION:
        render_ingestion_page()
    else:
        render_insights_page()
    render_footer()


if __name__ == "__main__":
    main()