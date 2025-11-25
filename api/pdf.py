# api/pdf.py — ONLY the 11-page PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image as RLImage, PageBreak, Spacer
from reportlab.platypus.paragraph import ParagraphStyle
from io import BytesIO
import base64
import json

def handler(event, context):
    data = json.loads(event["body"])
    
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.8*inch, bottomMargin=0.8*inch,
                            leftMargin=0.7*inch, rightMargin=0.7*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleBig', fontSize=32, leading=38, alignment=1, spaceAfter=20, textColor=colors.HexColor('#00dbde')))
    styles.add(ParagraphStyle(name='Subtitle', fontSize=16, leading=20, alignment=1, spaceAfter=40, textColor=colors.HexColor('#fc00ff')))
    styles.add(ParagraphStyle(name='Date', fontSize=12, alignment=1, textColor=colors.white))
    story = []

    story.append(Spacer(1, 3*inch))
    story.append(Paragraph("PRO FORMA AI", styles['TitleBig']))
    story.append(Paragraph("Institutional Investment Memorandum", styles['Subtitle']))
    story.append(Paragraph(f"Generated {data.get('date', 'November 2025')}", styles['Date']))
    story.append(PageBreak())

    # Key metrics table
    story.append(Paragraph("KEY RETURNS & METRICS", styles["Heading1"]))
    metric_data = [
        ["Base Equity IRR", data["base_irr"]],
        ["Median IRR (Monte Carlo)", data["p50"]],
        ["95th Percentile IRR", data["p95"]],
        ["Minimum DSCR", data["min_dscr"]],
        ["Equity Multiple", data["equity_multiple"]]
    ]
    t = Table(metric_data, colWidths=[4.8*inch, 2.2*inch])
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1e1e2e')),
                           ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#00dbde')),
                           ('GRID', (0,0), (-1,-1), 1, colors.white),
                           ('FONTSIZE', (0,0), (-1,-1), 14)]))
    story.append(t)
    story.append(PageBreak())

    # Images (sent as base64 from app.py)
    for title, img_b64 in [("EQUITY WATERFALL", data["waterfall_png"]),
                           ("SENSITIVITY ANALYSIS", data["sens_png"]),
                           ("DEBT SERVICE COVERAGE RATIO", data["dscr_png"]),
                           ("MONTE CARLO SIMULATION", data["irr_png"])]:
        story.append(Paragraph(title, styles["Heading1"]))
        img_data = base64.b64decode(img_b64)
        story.append(RLImage(BytesIO(img_data), width=7.5*inch, height=4.5*inch))
        story.append(PageBreak())

    # Cash flow table (sent as list of lists)
    story.append(Paragraph("FULL CASH FLOW & DSCR SCHEDULE", styles["Heading1"]))
    table_data = data["cf_table"]
    cf_table_obj = Table(table_data, colWidths=[0.7*inch]*len(table_data[0]))
    cf_table_obj.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00dbde')),
        ('TEXTCOLOR', (0,0), (-1,0), 'black'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.gray),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#1e1e2e')),
        ('TEXTCOLOR', (0,1), (-1,-1), 'white'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    story.append(cf_table_obj)
    story.append(PageBreak())

    story.append(Paragraph("CONFIDENTIAL — PRO FORMA AI", styles["Title"]))
    story.append(Paragraph("The model that closed $4.3B in 2025", styles["Italic"]))

    doc.build(story)
    pdf_buffer.seek(0)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/pdf"},
        "body": pdf_buffer.getvalue(),
        "isBase64Encoded": True
    }
