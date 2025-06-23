# JustUse Project: Asynchronous Scrum Log Guidelines

This document explains the purpose, structure, and usage of our team's Scrum Log, which serves as a replacement for traditional daily standup meetings.

## 1. Purpose: Why we use a Scrum Log

Our team is decentralized and operates asynchronously, meaning we don't all work at the same time or from the same location. To ensure continuous collaboration, transparency, and quick identification of issues, we use this Scrum Log instead of synchronous daily meetings.

The core goals of this log are to:

* **Foster Transparency:** Provide a clear, up-to-date overview of what each team member is working on, what they've accomplished, and any challenges they face.
* **Facilitate Synchronization:** Allow team members to quickly understand the current status of the Sprint, without needing to coordinate a specific meeting time.
* **Identify Impediments:** Act as an early warning system for blockers or open questions that need attention from the Scrum Master or other team members.
* **Support Asynchronous Work:** Enable team members to update their status and review others' progress at their convenience, fitting into diverse schedules.
* **Maintain an Audit Trail:** Leverage Git history to track contributions, progress, and decisions over time.

## 2. Structure: How the Scrum Log is Organized

Each **Sprint** (which currently spans **one calendar week, Monday to Sunday**) will have its own dedicated Markdown file.

* **Location:** All Scrum Log files are located in this directory: `/docs/scrum/`.
* **Naming Convention:** Files are named `YYYY-WW.md`, where `YYYY` is the year and `WW` is the week number (e.g., `2025-26.md` for week 26 of 2025).

Inside each weekly log file, entries are organized by day:

* **Daily Headings:** Each day of the Sprint begins with a `## [Day of Week]` heading (e.g., `## Monday`).
* **Core Sections:** Under each daily heading, entries are grouped into specific categories:
    * **`### Done since last update:`**
        * List all tasks or progress completed since your last entry.
        * Focus on completed work that contributes to the **Sprint Goal**.
        * Use checkboxes (`[x]`) for clarity.
    * **`### Next up / To do:`**
        * Outline the tasks you plan to work on next, ideally contributing to the **Sprint Goal**.
        * Be specific about your immediate next steps.
        * Use checkboxes (`[ ]`) to indicate tasks that are not yet started.
    * **`### Impediments / Blockers:`**
        * **Crucial for the Scrum Master!** Clearly state any obstacles, technical issues, or missing information that are preventing you from making progress.
        * If no blockers, explicitly state "None at the moment."
    * **`### Decisions Made:`**
        * Document any significant decisions or agreements reached regarding tasks, implementation approaches, or priorities. This provides context for the work being done.
    * **`### Open Questions:`**
        * List any questions for other team members, the Product Owner, or the Scrum Master. This is a direct call for input or discussion.

## 3. Usage: How to Contribute and Interact

1.  **Pull Before You Edit:** Always `git pull origin main` (or your main development branch) before making an update to ensure you have the latest version of the log. This minimizes merge conflicts.
2.  **Append Your Entry:** Add your updates to the current day's section at the bottom of the active Sprint Log file.
3.  **Commit Your Changes:**
    * After adding your entry, commit the change immediately.
    * Use a clear commit message, e.g., `docs: Scrum Log update`.
4.  **Push Your Changes:** `git push origin main`  to share your update with the team.
5.  **Review Daily:** Make it a habit to **read the entire log file daily** (or at the start of your work session) to stay informed about what others are doing, their progress, and crucially, any new **impediments** or **open questions**. Pay attention to Git diffs for quick scanning.
6.  **Resolve Issues:** If you see an impediment or open question that you can help with, reach out directly to the team member (e.g., via chat or a quick call) to resolve it. The Scrum Master will also be monitoring these actively.
7.  **Be Concise:** While detailed, try to be clear and to the point - one line per item is ideal.

## 4. Key Considerations for Async Teams

* **Git is our "Who" and "When":** We intentionally omit author names and exact timestamps within the log content itself. This is because Git's commit history (`git log`, `git blame`, GitHub/GitLab diffs) already provides precise information on who made which changes and when. This reduces visual clutter in the Markdown file.
* **Communication Beyond the Log:** While the log is central, it's not the *only* form of communication. Complex discussions or urgent issues should still be handled via chat, video calls, or dedicated issue tracking (e.g., GitHub Issues). The log helps identify when these deeper discussions are needed.

By adhering to these guidelines, we ensure our asynchronous Scrum Log effectively supports our collaborative development of JustUse.