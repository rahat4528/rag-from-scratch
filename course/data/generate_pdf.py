"""Script to generate course/data/policy_brief.pdf for document loader practice."""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_policy_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=8
    )

    story = [
        Paragraph("Novasphere Dynamics - Autonomous Robotics Safety & Deployment Brief", title_style),
        Paragraph("Document ID: SAF-2026-B | Classification: Confidential Internal", body_style),
        Spacer(1, 10),
        Paragraph("1. Purpose & Scope", h2_style),
        Paragraph("This technical brief specifies mandatory field safety protocols for deploying autonomous mobile robots (AMRs) and industrial drones connected to the Novasphere OS control plane.", body_style),
        Paragraph("2. Physical Safety Perimeter & E-Stop Requirements", h2_style),
        Paragraph("During active commissioning, all human personnel must maintain a minimum 3-meter physical safety perimeter around active AMR testing zones. All robotic units must be outfitted with a dual-channel hardware Emergency Stop (E-Stop) button rated at SIL-3 / PLe. In the event of network disconnection exceeding 40ms, ZeroGravity failover or automatic safe-braking will engage.", body_style),
        Paragraph("3. Radio Frequency & Telemetry Prerequisites", h2_style),
        Paragraph("Industrial deployment facilities must provide dedicated Wi-Fi 7 or private 5G coverage on UDP port 9872. Minimum signal strength must exceed -65 dBm across all operating corridors. If signal drops below -75 dBm, the node must decelerate to a crawl speed of 0.2 m/s.", body_style),
        Paragraph("4. Incident Reporting & Escalation", h2_style),
        Paragraph("Any safety near-miss or hardware anomaly must be logged within 60 minutes via the internal Safety Portal. For critical Severity-1 emergencies involving physical damage, contact the 24/7 Operations Hotline immediately at +41 44 888 0199.", body_style)
    ]

    doc.build(story)
    print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    create_policy_pdf(str(out_dir / "policy_brief.pdf"))
