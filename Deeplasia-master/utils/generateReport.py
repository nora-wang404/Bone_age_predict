import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle, Paragraph, Image
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def generate_bone_age_report(records, filename_time, report_title, font_path="simhei.ttf"):
    """
    Generate bone age prediction report PDF (text and images on separate lines, images centered)
    Parameters:
    records: List of records, each element is (original image path, heatmap path, time string, bone age value)
    filename_time: Timestamp used to generate output filename
    report_title: Main title of the report
    font_path: Path to Chinese font file
    """
    # Ensure font file exists
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font file {font_path} does not exist, please check the path")

    # Create PDF file
    output_path = f"./upload/{filename_time}_BoneAgePredictionReport.pdf"
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter  # Page width and height

    # Register Chinese font
    try:
        font_name = "SimSun"
        pdfmetrics.registerFont(TTFont(font_name, font_path))
    except Exception as e:
        raise RuntimeError(f"Font registration failed: {str(e)}")

    # Custom styles
    styles = getSampleStyleSheet()

    # Title style
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontName=font_name,
        fontSize=20,
        alignment=TA_CENTER,
        spaceAfter=12
    )

    # Normal text style (left-aligned)
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=12,
        leading=16,
        spaceAfter=4,
        alignment=TA_LEFT
    )

    # Draw title
    title = Paragraph(report_title, title_style)
    title.wrapOn(c, width - 1.2*inch, height)
    title.drawOn(c, inch, height - 1*inch)  # Title position

    # Separator line below title
    c.setLineWidth(1)
    c.line(inch, height - 1.2*inch, width - inch, height - 1.2*inch)

    # Content area starting position
    y_position = height - 1.5*inch  # Space below title
    records_per_page = 1  # Display 1 record per page (due to large images)
    current_record = 0
    page_width = width

    # Process each record
    while current_record < len(records):
        for _ in range(records_per_page):
            if current_record >= len(records):
                break

            # Parse record data
            img_path, heatmap_path, timestamp, bone_age = records[current_record]

            try:
                # Process original image (proportionally scaled, max width 80% of page)
                img = Image(img_path)
                max_img_width = page_width * 0.8  # Maximum image width (80% of page)
                max_img_height = 3.5*inch         # Maximum image height
                img_ratio = img.drawWidth / img.drawHeight

                if img.drawWidth > max_img_width:
                    img.drawWidth = max_img_width
                    img.drawHeight = max_img_width / img_ratio
                if img.drawHeight > max_img_height:
                    img.drawHeight = max_img_height
                    img.drawWidth = max_img_height * img_ratio

                # Process heatmap (same width as original image)
                heatmap = Image(heatmap_path)
                heatmap_ratio = heatmap.drawWidth / heatmap.drawHeight
                heatmap.drawWidth = img.drawWidth  # Heatmap same width as original image
                heatmap.drawHeight = heatmap.drawWidth / heatmap_ratio

                # Construct record content (text on separate lines, images on separate lines)
                data = [
                    # Time information (separate line)
                    [Paragraph(f"Detection time: {timestamp}", normal_style)],
                    # Bone age result (separate line)
                    [Paragraph(f"Predicted bone age: {bone_age} months", normal_style)],
                    # Original image title (separate line)
                    [Paragraph("Original image:", normal_style)],
                    # Original image (separate line, centered)
                    [img],
                    # Heatmap title (separate line)
                    [Paragraph("Model attention area (heatmap):", normal_style)],
                    # Heatmap (separate line, centered)
                    [heatmap]
                ]

                # Create table (single column layout, entire column centered)
                table = Table(data, colWidths=[page_width - 2*inch])  # Column width = page width minus left and right margins
                table_style = TableStyle([
                    # All cells vertically top-aligned
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    # Text lines left-aligned, image lines centered
                    ('ALIGN', (0, 0), (0, 1), 'LEFT'),    # Time and bone age lines left-aligned
                    ('ALIGN', (0, 2), (0, 2), 'LEFT'),    # Original image title left-aligned
                    ('ALIGN', (0, 3), (0, 3), 'CENTER'),  # Original image centered
                    ('ALIGN', (0, 4), (0, 4), 'LEFT'),    # Heatmap title left-aligned
                    ('ALIGN', (0, 5), (0, 5), 'CENTER'),  # Heatmap centered
                    # Line spacing settings
                    ('BOTTOMPADDING', (0, 0), (0, 1), 10),  # Bottom padding for text lines
                    ('BOTTOMPADDING', (0, 2), (0, 4), 6),   # Bottom padding for image titles
                    ('BOTTOMPADDING', (0, 3), (0, 5), 12),  # Bottom padding for images
                ])
                table.setStyle(table_style)

                # Calculate table height
                tbl_width, tbl_height = table.wrapOn(c, page_width - 2*inch, y_position)

                # Page break check (leave 1.5 inch margin at bottom)
                if y_position - tbl_height < 0.5*inch:
                    c.showPage()  # Create new page
                    # Draw title and separator line on new page
                    title.drawOn(c, inch, height - 1*inch)
                    c.line(inch, height - 1.2*inch, width - inch, height - 1.2*inch)
                    y_position = height - 1.2*inch  # Reset starting position

                # Draw table
                table.drawOn(c, inch, y_position - tbl_height)
                # Update starting position for next record (reserve space between records)
                y_position -= tbl_height + 1*inch

                current_record += 1

            except Exception as e:
                print(f"Error processing record {current_record}: {str(e)}")
                current_record += 1
                continue

        # Check if page break is needed
        if current_record < len(records):
            c.showPage()
            # Draw title and separator line on new page
            title.drawOn(c, inch, height - 1*inch)
            c.line(inch, height - 1.2*inch, width - inch, height - 1.2*inch)
            y_position = height - 1.8*inch  # Reset starting position

    # Save PDF
    c.save()
    print(f"Bone age prediction report generated: {output_path}")
if __name__ == "__main__":

    test_records = [
        (
            "../upload/7256.png",
            "../upload/heat_7256.png",
            "2025-07-20 14:08:10",
            "75.1"
        )
    ]
    generate_bone_age_report(
        records=test_records,
        filename_time="20250720_report",
        report_title="骨龄预测分析报告",
        font_path="./SimSun.ttf"  
    )