"""
interventions.py
────────────────────────────────────────────────────────────
Rule-based intervention recommendation engine.
Given a student's predicted risk level and input features,
generates specific, actionable intervention suggestions.

This is the "real-world use case" component that turns a
prediction into something an academic advisor can act on.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Intervention:
    category:    str
    priority:    str          # "Critical" | "High" | "Medium" | "Low"
    action:      str
    responsible: str          # who acts on this
    timeline:    str


def get_interventions(student: dict, risk_label: str) -> List[Intervention]:
    """
    Generate personalised interventions based on student profile.

    Parameters
    ----------
    student    : dict of raw (pre-processed) student features
    risk_label : "Low Risk" | "Medium Risk" | "High Risk"

    Returns
    -------
    List[Intervention] sorted by priority
    """
    interventions = []
    priority_map = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

    # ── Attendance issues ─────────────────────────────────
    att = student.get("attendance_pct", 100)
    if att < 50:
        interventions.append(Intervention(
            category="Attendance",
            priority="Critical",
            action=f"Attendance at {att:.0f}% — immediate counselling session required. "
                   "Investigate root cause (transport, health, family, financial).",
            responsible="Academic Advisor",
            timeline="This week"
        ))
    elif att < 65:
        interventions.append(Intervention(
            category="Attendance",
            priority="High",
            action=f"Attendance at {att:.0f}% — send written warning. "
                   "Schedule bi-weekly check-ins. Connect with transport/hostel office if needed.",
            responsible="Class Advisor",
            timeline="Within 3 days"
        ))
    elif att < 75:
        interventions.append(Intervention(
            category="Attendance",
            priority="Medium",
            action=f"Attendance at {att:.0f}% — monitor and send automated reminder. "
                   "Discuss time management strategies.",
            responsible="Department Coordinator",
            timeline="Within 2 weeks"
        ))

    # ── Academic performance ───────────────────────────────
    midterm = student.get("midterm_score", 100)
    gpa     = student.get("prev_gpa", 4.0)

    if midterm < 35 or gpa < 1.5:
        interventions.append(Intervention(
            category="Academic Support",
            priority="Critical",
            action=f"Midterm score {midterm:.0f}/100, GPA {gpa:.2f} — enrol in intensive "
                   "academic recovery programme. Assign peer tutor immediately. "
                   "Notify parents/guardian.",
            responsible="Department Head",
            timeline="Immediately"
        ))
    elif midterm < 50 or gpa < 2.0:
        interventions.append(Intervention(
            category="Academic Support",
            priority="High",
            action=f"Midterm score {midterm:.0f}/100, GPA {gpa:.2f} — schedule subject-specific "
                   "tutoring sessions. Provide practice test banks. Review study strategy.",
            responsible="Subject Teacher + Advisor",
            timeline="Within 1 week"
        ))
    elif midterm < 60:
        interventions.append(Intervention(
            category="Academic Support",
            priority="Medium",
            action=f"Midterm score {midterm:.0f}/100 — recommend supplemental study resources. "
                   "Encourage study group participation.",
            responsible="Subject Teacher",
            timeline="Within 2 weeks"
        ))

    # ── Failed subjects / backlogs ─────────────────────────
    failed = student.get("failed_subjects", 0)
    if failed >= 3:
        interventions.append(Intervention(
            category="Academic Planning",
            priority="Critical",
            action=f"{failed} failed subjects — refer to academic probation committee. "
                   "Create personalised academic recovery plan. Consider course load reduction.",
            responsible="Academic Committee",
            timeline="Immediately"
        ))
    elif failed >= 1:
        interventions.append(Intervention(
            category="Academic Planning",
            priority="High",
            action=f"{failed} failed/backlog subject(s) — schedule supplementary exam preparation. "
                   "Weekly progress tracking with advisor.",
            responsible="Academic Advisor",
            timeline="Within 1 week"
        ))

    # ── Study habits ──────────────────────────────────────
    study_hrs = student.get("study_hours_day", 3)
    if study_hrs is None:
        study_hrs = 3
    if study_hrs < 1.5:
        interventions.append(Intervention(
            category="Study Habits",
            priority="High",
            action=f"Only {study_hrs:.1f} hrs/day of study — conduct time management workshop. "
                   "Introduce Pomodoro technique. Create personalised study schedule.",
            responsible="Student Counsellor",
            timeline="Within 1 week"
        ))
    elif study_hrs < 2.5:
        interventions.append(Intervention(
            category="Study Habits",
            priority="Medium",
            action=f"{study_hrs:.1f} hrs/day study time — share structured study plan templates. "
                   "Recommend study apps (Notion, Forest).",
            responsible="Class Advisor",
            timeline="Within 2 weeks"
        ))

    # ── Psychosocial: Stress ──────────────────────────────
    stress = student.get("stress_level", 5)
    motivation = student.get("motivation", 5)
    if stress >= 8:
        interventions.append(Intervention(
            category="Mental Health",
            priority="Critical",
            action=f"Stress level {stress}/10 — immediate referral to campus counsellor. "
                   "Assess for burnout or anxiety. Reduce academic pressure temporarily if needed.",
            responsible="Welfare Officer + Counsellor",
            timeline="This week"
        ))
    elif stress >= 6:
        interventions.append(Intervention(
            category="Mental Health",
            priority="High",
            action=f"Stress level {stress}/10 — invite to stress management workshop. "
                   "Encourage mindfulness practice. Follow up monthly.",
            responsible="Student Counsellor",
            timeline="Within 2 weeks"
        ))

    if motivation <= 3:
        interventions.append(Intervention(
            category="Motivation",
            priority="High",
            action=f"Motivation score {motivation}/10 — career counselling session. "
                   "Connect with alumni mentors. Identify and reconnect student with academic goals.",
            responsible="Career Centre",
            timeline="Within 1 week"
        ))

    # ── Financial concerns ────────────────────────────────
    income = student.get("family_income", "Middle")
    scholarship = student.get("scholarship", 0)
    if income == "Low" and not scholarship:
        interventions.append(Intervention(
            category="Financial Support",
            priority="High",
            action="Student from low-income background with no scholarship — refer to financial aid "
                   "office. Assist with scholarship/bursary application. Connect with HEC need-based aid.",
            responsible="Financial Aid Office",
            timeline="Within 1 week"
        ))

    # ── Part-time job impact ───────────────────────────────
    if student.get("part_time_job", 0) and study_hrs < 2:
        interventions.append(Intervention(
            category="Work-Study Balance",
            priority="Medium",
            action="Part-time employment combined with low study hours. Discuss flexible schedule "
                   "options. Share on-campus job opportunities with lighter time commitment.",
            responsible="Student Affairs",
            timeline="Within 2 weeks"
        ))

    # ── Assignment submission ──────────────────────────────
    assign_pct = student.get("assignment_sub_pct", 100)
    if assign_pct < 50:
        interventions.append(Intervention(
            category="Academic Engagement",
            priority="High",
            action=f"Only {assign_pct:.0f}% assignments submitted — identify barriers (internet "
                   "access, understanding, time). Provide deadline extensions if warranted.",
            responsible="Subject Teacher",
            timeline="Immediately"
        ))

    # ── If Low Risk — preventive ───────────────────────────
    if risk_label == "Low Risk" and not interventions:
        interventions.append(Intervention(
            category="Enrichment",
            priority="Low",
            action="Student performing well. Encourage participation in research projects, "
                   "hackathons, and extracurricular tech activities to strengthen portfolio.",
            responsible="Department Faculty",
            timeline="Ongoing"
        ))

    # Sort by priority
    interventions.sort(key=lambda x: priority_map.get(x.priority, 99))
    return interventions


def format_interventions_text(interventions: List[Intervention]) -> str:
    """Format interventions for display."""
    lines = []
    for i, inv in enumerate(interventions, 1):
        lines.append(
            f"\n[{i}] [{inv.priority}] {inv.category}\n"
            f"    Action      : {inv.action}\n"
            f"    Responsible : {inv.responsible}\n"
            f"    Timeline    : {inv.timeline}"
        )
    return "\n".join(lines)
