"""Generate a small sample PDF so the RAG pipeline can be tested end-to-end."""

import os

from fpdf import FPDF

from config import DATA_DIR


def main(model: str = "cli_small_24b") -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "sample_report.pdf")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "Q3 2026 Product Report", ln=True)
    pdf.ln(4)

    blocks = [
        ("Revenue",
         "Total revenue reached $4.2M in Q3 2026, up 23% quarter over quarter. "
         "Subscription revenue contributed $3.1M, while professional services added $1.1M."),
        ("Users",
         "Monthly active users grew to 128,000. Customer retention improved to 94% after "
         "the guided-onboarding rollout in August 2026."),
        ("AI Features",
         "The new AI chat assistant was launched in beta in September 2026. Early metrics "
         "show a 35% reduction in support tickets and 4.8/5 average satisfaction."),
        ("Risks",
         "Infrastructure costs rose 18% due to GPUs used by the AI features. Management "
         "expects unit economics to improve once batching and quantization go live in Q4."),
        ("Outlook",
         "Guidance for Q4 2026 is $5.0M revenue. The team plans to double down on "
         "enterprise segments and expand the AI assistant to mobile platforms."),
    ]
    for title, body in blocks:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, title, ln=True)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, body)
        pdf.ln(2)

    pdf.output(path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()