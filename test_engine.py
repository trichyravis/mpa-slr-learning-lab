import unittest
import numpy as np
import pandas as pd
from scipy import stats
from engine import fit_slr,coefficient_table,predict,residual_table,validate_pairs,diagnostics,holdout,descriptive_table
from samples import sample_data,SAMPLE_NAMES

class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.x=np.arange(1,31,dtype=float)
        self.y=12+1.7*self.x+np.random.default_rng(42).normal(0,3,len(self.x))
        self.fit=fit_slr(self.x,self.y)

    def test_independent_scipy_reference(self):
        ref=stats.linregress(self.x,self.y);c=coefficient_table(self.fit)
        np.testing.assert_allclose(self.fit.beta,[ref.intercept,ref.slope],rtol=1e-12)
        self.assertAlmostEqual(c.loc[1,'Standard error'],ref.stderr,places=12)
        self.assertAlmostEqual(c.loc[1,'p-value'],ref.pvalue,places=12)
        self.assertAlmostEqual(self.fit.r2,ref.rvalue**2,places=12)

    def test_anova_identity_and_f_t(self):
        f=self.fit;c=coefficient_table(f)
        self.assertAlmostEqual(f.ssr+f.sse,f.sst,places=8)
        self.assertAlmostEqual(f.f,c.loc[1,'t statistic']**2,places=8)
        self.assertAlmostEqual(f.f_p,c.loc[1,'p-value'],places=12)
        self.assertAlmostEqual(f.residual.mean(),0,places=12)
        self.assertAlmostEqual(f.x@f.residual,0,places=8)

    def test_prediction_interval_formula(self):
        f=self.fit;p=predict(f,[f.x.mean(),f.x.max()+1]);critical=stats.t.ppf(.975,f.df)
        self.assertAlmostEqual(p.loc[0,'Mean CI upper']-p.loc[0,'Predicted Y'],critical*np.sqrt(f.mse/f.n),places=9)
        self.assertGreater(p.loc[0,'Individual PI upper'],p.loc[0,'Mean CI upper'])
        self.assertFalse(p.loc[0,'Extrapolation']);self.assertTrue(p.loc[1,'Extrapolation'])
        self.assertNotIn('Individual PI upper',predict(f,[12],'HC3'))

    def test_hc3_direct_reference(self):
        f=self.fit;X=np.column_stack([np.ones(f.n),f.x]);inv=np.linalg.inv(X.T@X)
        expected=inv@(X.T@np.diag((f.residual/(1-f.leverage))**2)@X)@inv
        np.testing.assert_allclose(f.hc3_covariance,expected,rtol=1e-10,atol=1e-10)

    def test_hac_direct_reference(self):
        f=fit_slr(self.x,self.y,hac_lags=2);X=np.column_stack([np.ones(f.n),f.x]);inv=np.linalg.inv(X.T@X)
        scores=X*f.residual[:,None];meat=scores.T@scores
        for lag in [1,2]:
            g=scores[lag:].T@scores[:-lag];meat+=(1-lag/3)*(g+g.T)
        np.testing.assert_allclose(f.hac_covariance,f.n/f.df*inv@meat@inv,rtol=1e-10)

    def test_press_actual_leave_one_out(self):
        expected=[]
        for i in range(len(self.x)):
            keep=np.arange(len(self.x))!=i;ref=stats.linregress(self.x[keep],self.y[keep])
            expected.append(self.y[i]-ref.intercept-ref.slope*self.x[i])
        np.testing.assert_allclose(residual_table(self.fit)['LOOCV error'],expected,atol=1e-10)

    def test_official_sample(self):
        d,x,y,*_=sample_data('GDP and PFCE — official snapshot');f=fit_slr(d[x],d[y])
        self.assertEqual(f.n,16);self.assertAlmostEqual(f.beta[1],1.5986811453452658,places=10)
        self.assertAlmostEqual(f.beta[0],741400.7673493757,places=5)
        self.assertAlmostEqual(f.r2,.9517213431494038,places=12)

    def test_validation_no_silent_zero_imputation(self):
        d=pd.DataFrame({'X':[0,-1,'2',None,'oops',np.inf,True],'Y':[1,2,3,4,5,6,7]})
        valid,bad=validate_pairs(d,'X','Y');self.assertEqual(list(valid.X),[0,-1,2]);self.assertEqual(len(bad),4)

    def test_degenerate_inputs(self):
        for x,y in [([1]*5,[1,2,3,4,5]),([1,2,3,4,5],[1]*5),([1,2,3],[2,3,4]),([1,2,3,4],[2,4,6,8])]:
            with self.assertRaises(ValueError):fit_slr(x,y)
        with self.assertRaises(ValueError):fit_slr([1,2,3,4],[1,2,np.nan,5])

    def test_holdout_no_leakage(self):
        m,rows=holdout(self.x,self.y,ordered=True);count=int(np.ceil(len(self.x)*.25))
        ref=stats.linregress(self.x[:-count],self.y[:-count])
        np.testing.assert_allclose(rows['Predicted Y'],ref.intercept+ref.slope*self.x[-count:])
        self.assertEqual(m['Training n']+m['Test n'],30)

    def test_all_samples(self):
        for name in SAMPLE_NAMES:
            d,x,y,*_=sample_data(name);f=fit_slr(d[x],d[y]);self.assertGreater(len(diagnostics(f,True)),3)
            self.assertFalse(coefficient_table(f).select_dtypes('number').isna().any().any())
            self.assertEqual(len(descriptive_table(f.x,f.y)),2)

if __name__=='__main__':unittest.main()
