import unittest
from io import BytesIO
import pandas as pd
from streamlit.testing.v1 import AppTest
from openpyxl import load_workbook
from exports import excel_bytes,report_html,TEMPLATE_INSTRUCTIONS

class AppTests(unittest.TestCase):
    def test_navigation_and_inference(self):
        app=AppTest.from_file('app.py',default_timeout=60).run()
        self.assertEqual(len(app.exception),0)
        for label in ['Data & descriptives','Regression & ANOVA','Assumptions & diagnostics','Prediction & validation','Learning guide','Download report']:
            app.sidebar.radio[1].set_value(label).run();self.assertEqual(len(app.exception),0,label)
        app.sidebar.selectbox[1].set_value('HC3').run();self.assertEqual(len(app.exception),0)
        app.sidebar.selectbox[0].set_value('GDP and PFCE — official snapshot').run();self.assertEqual(len(app.exception),0)
        app.sidebar.selectbox[1].set_value('HAC').run();self.assertEqual(len(app.exception),0)

    def test_excel_roundtrip_and_formula_safety(self):
        frame=pd.DataFrame({'Observation':['=1+1','Case 2'],'X':[1,2],'Y':[2,4]})
        content=excel_bytes({'Data':frame},TEMPLATE_INSTRUCTIONS);wb=load_workbook(BytesIO(content),data_only=False)
        self.assertEqual(wb['Data']['A2'].data_type,'s');self.assertEqual(wb['Data']['B2'].value,1)
        self.assertEqual(wb['Data'].freeze_panes,'A2')
        self.assertEqual(pd.read_excel(BytesIO(content),sheet_name='Data').X.tolist(),[1,2])
        report=report_html({'Data':frame},{'X':'<script>'});self.assertIn('&lt;script&gt;',report)

if __name__=='__main__':unittest.main()
