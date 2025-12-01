"""Create sample travel policy document"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config


def create_sample_policy():
    """Create a sample travel policy PDF"""

    output_path = Config.POLICY_DIR / "company_travel_policy.pdf"

    # Create PDF
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )

    # Container for the 'Flowable' objects
    elements = []

    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor="#1f4788",
        spaceAfter=30,
        alignment=1,  # Center
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor="#1f4788",
        spaceAfter=12,
        spaceBefore=12,
    )

    # Title
    title = Paragraph("Company Travel Policy", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2 * inch))

    # Policy content
    content = [
        (
            "1. Flight Booking Policy",
            [
                "• Domestic flights: Economy class only, maximum budget of $500 per ticket",
                "• International flights: Business class allowed for flights over 6 hours, maximum budget of $2,500",
                "• Flights must be booked at least 2 weeks in advance for best rates",
                "• Direct flights preferred; connecting flights only if cost savings exceed 20%",
                "• Preferred airlines: Delta, United, American Airlines for domestic; Emirates, Lufthansa for international",
            ],
        ),
        (
            "2. Ground Transportation Policy",
            [
                "• Airport transfers: Taxi or ride-share services up to $75 per trip",
                "• Daily cab usage: Maximum $100 per day for business purposes",
                "• Rental cars: Compact or mid-size vehicles only, maximum $60 per day",
                "• Public transportation encouraged when available and safe",
                "• Mileage reimbursement for personal vehicles: $0.58 per mile",
            ],
        ),
        (
            "3. Accommodation Policy",
            [
                "• Hotel budget: Up to $200 per night in major cities, $150 in other locations",
                "• Preferred hotel chains: Marriott, Hilton, Hyatt",
                "• Standard room only; suite upgrades not reimbursed unless required for business meetings",
                "• Extended stays (7+ nights): Serviced apartments up to $175 per night",
            ],
        ),
        (
            "4. Meal and Per Diem Allowances",
            [
                "• Breakfast: $15 per day",
                "• Lunch: $25 per day",
                "• Dinner: $40 per day",
                "• Total daily meal allowance: $80",
                "• Business meals with clients: Up to $150 per person with prior approval",
            ],
        ),
        (
            "5. Travel Budget Limits",
            [
                "• Domestic trips (3 days): Total budget up to $1,500 including flights, accommodation, and ground transport",
                "• International trips (5 days): Total budget up to $4,000 including flights, accommodation, and ground transport",
                "• Conference attendance: Additional $500 for registration and related expenses",
            ],
        ),
        (
            "6. Booking Guidelines",
            [
                "• All travel must be booked through approved channels",
                "• Travel dates and times should minimize impact on work schedule",
                "• Weekend stays: Allowed if it results in lower airfare (minimum 20% savings)",
                "• Travel insurance: Recommended for international trips, covered up to $100",
            ],
        ),
        (
            "7. Additional Benefits",
            [
                "• Airport lounge access: Provided for flights over 4 hours or international travel",
                "• Wi-Fi costs: Reimbursed for in-flight and hotel internet access",
                "• Baggage fees: One checked bag allowed for trips over 3 days",
                "• Travel accessories: Up to $50 reimbursement for necessary travel items per trip",
            ],
        ),
    ]

    # Add content to PDF
    for heading, items in content:
        elements.append(Paragraph(heading, heading_style))
        for item in items:
            elements.append(Paragraph(item, styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))
        elements.append(Spacer(1, 0.2 * inch))

    # Footer
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"], fontSize=10, textColor="#666666", alignment=1
    )
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("Policy effective date: January 1, 2025", footer_style))
    elements.append(
        Paragraph("For questions, contact: travel@company.com", footer_style)
    )

    # Build PDF
    doc.build(elements)

    print(f"✅ Sample travel policy created at: {output_path}")


if __name__ == "__main__":
    try:
        create_sample_policy()
    except ImportError:
        print("⚠️  reportlab not installed. Installing...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
        create_sample_policy()
