from io import BytesIO
from datetime import datetime, timezone
import html
import re
import pandas as pd
from content import ASSUMPTIONS, LIMITATIONS


def excel_bytes(tables, instructions=None):
    """Formatted Excel output; strings are explicitly text to prevent formula injection."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.worksheet.table import Table, TableStyleInfo
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    wb.remove(wb.active)
    if instructions:
        tables = {**tables, "Instructions": pd.DataFrame({"Guidance": instructions})}
    for j,(name,frame) in enumerate(tables.items()):
        ws = wb.create_sheet(re.sub(r"[\\/*?:\[\]]", "", name)[:31])
        ws.sheet_view.showGridLines=False
        ws.freeze_panes="A2"
        ws.append([str(c) for c in frame.columns])
        for cell in ws[1]:
            cell.fill=PatternFill("solid", fgColor="142D4E")
            cell.font=Font(name="Calibri",size=13,bold=True,color="FFFFFF")
            cell.alignment=Alignment(vertical="center",wrap_text=True)
        ws.row_dimensions[1].height=32
        for row in frame.itertuples(index=False,name=None):
            vals=[]
            for value in row:
                if pd.isna(value): value=None
                elif hasattr(value,"item"): value=value.item()
                vals.append(value)
            ws.append(vals)
            for cell in ws[ws.max_row]:
                if isinstance(cell.value,str): cell.data_type="s"
                cell.font=Font(name="Calibri",size=12,color="174C83" if name=="Data" else "233B53")
                cell.alignment=Alignment(vertical="top",wrap_text=True)
                if isinstance(cell.value,(float,int)) and not isinstance(cell.value,bool):
                    cell.number_format="0.0000E+00" if "p-value" in str(ws.cell(1,cell.column).value) else "#,##0.0000"
            ws.row_dimensions[ws.max_row].height=28 if len(frame.columns)>1 else 56
        if len(frame) and len(frame.columns)>1:
            tab=Table(displayName=f"SLRTable{j+1}",ref=f"A1:{get_column_letter(len(frame.columns))}{ws.max_row}")
            tab.tableStyleInfo=TableStyleInfo(name="TableStyleMedium2",showRowStripes=True)
            ws.add_table(tab)
        for i,col in enumerate(frame.columns,1):
            length=max([len(str(col))]+[len(str(v)) for v in frame[col].head(100)])
            ws.column_dimensions[get_column_letter(i)].width=min(65,max(20,length+3))
    output=BytesIO()
    wb.save(output)
    return output.getvalue()


TEMPLATE_INSTRUCTIONS = [
    "SLR Learning Lab — The Mountain Path Academy",
    "On Data, keep one header row in row 1. Each subsequent row is one paired observation.",
    "Replace the sample with your data. Any descriptive column names are accepted. Add an optional Observation ID or date column.",
    "X is the explanatory variable; Y is the response. Use numeric Excel cells without currency symbols or unit labels in the cells.",
    "Use consistent units. Record units, source and frequency in a separate note or Instructions sheet.",
    "For time series, place rows in chronological order, earliest first, at regular intervals. The app preserves your row order.",
    "Choose Data in the uploader, select X, Y and an optional label column. Extra columns are allowed and ignored unless selected.",
    "Missing/invalid/infinite numeric pairs are displayed for review; fitting waits for explicit approval to exclude them.",
    "Do not add totals, subtotals, merged headers or blank separator rows to the Data table. No automatic outlier removal or zero imputation.",
    "Excel formulas must have cached numerical results; otherwise paste values before upload. Do not upload passwords or sensitive identifiers.",
    "Sample data in this template are synthetic and illustrative, not observed business data.",
]


def report_html(tables, metadata):
    sections = "".join(f"<h2>{html.escape(k)}</h2>{v.to_html(index=False,escape=True,float_format=lambda x:f'{x:,.6g}')}" for k,v in tables.items())
    assumptions = pd.DataFrame(ASSUMPTIONS,columns=["Assumption","Meaning","Assessment","Consequence"])
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>SLR analysis report</title>
<style>body{{font:16px/1.5 system-ui;color:#16314d;max-width:1200px;margin:40px auto;padding:20px}}h1,h2{{color:#153752}}h1{{border-bottom:4px solid #C9A44C}}table{{border-collapse:collapse;font-size:12px;width:100%;margin:15px 0;display:block;overflow-x:auto}}td,th{{padding:9px;border:1px solid #d7e0e8;text-align:left}}th{{background:#eaf1f7}}.note{{background:#fff7df;padding:18px}}@media print{{body{{margin:0}}table{{display:table;font-size:9px}}h2{{break-after:avoid}}tr{{break-inside:avoid}}button{{display:none}}}}</style></head>
<body><p>THE MOUNTAIN PATH ACADEMY</p><h1>Simple Linear Regression</h1><button onclick='window.print()'>Print / Save as PDF</button>
<p>Created {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
{pd.DataFrame(metadata.items(),columns=['Setting','Value']).to_html(index=False,escape=True)}
<p class='note'>Association is not causation. All inference is conditional on the chosen model, sampling design and assumptions. Non-significant diagnostic results do not verify assumptions.</p>
{sections}<h2>Assumptions</h2>{assumptions.to_html(index=False,escape=True)}<h2>Limitations</h2><ul>
{''.join('<li>'+html.escape(s)+'</li>' for s in LIMITATIONS)}</ul></body></html>"""
