functions {
  real f_constrained(real f0, real gamma0) {
    real x = log(f0/gamma0);

    return f0*(0.97598764 + x*(0.06695086 + x*(-0.09355756 + x*(0.06990699 - x*0.02013998))));
  }

  real g_constrained(real f0, real gamma0) {
    real x = log(f0/gamma0);

    return gamma0*(3.02608227 + x*(-0.06722567 + x*(0.11394527 + x*(-0.10399184 + x*0.03334442))));
  }

  vector rd(vector t, real cos_inc, real A, real Fp, real Fc, real phi, real gamma, real f) {
    return exp(-t*gamma).*(0.5*Fp*A*(1+cos_inc*cos_inc)*cos(2*pi()*f*t + phi) + Fc*A*cos_inc*sin(2*pi()*f*t + phi));
  }
}

data {
  int nobs;
  int nsamp;

  real t0[nobs];
  vector[nsamp] ts[nobs];
  vector[nsamp] strain[nobs];
  matrix[nsamp,nsamp] L[nobs];

  /* For fundamental mode. */
  real mu_logf;
  real sigma_logf;

  /* For fundamental mode. */
  real mu_loggamma;
  real sigma_loggamma;

  vector[2] FpFc[nobs];

  real cos_inc;

  real Amax;

  real df_dg_max;
}

transformed data {
  int nmode = 2;
}

parameters {
  real<lower=0> f0;
  real<lower=exp(-1.546)*f0, upper=exp(0.399)*f0> gamma0;

  real<lower=-df_dg_max, upper=df_dg_max> df1;
  real<lower=-df_dg_max, upper=df_dg_max> dg1;

  vector<lower=0, upper=Amax>[nmode] A;
  unit_vector[2] xy_phase[nmode];
}

transformed parameters {
  vector[nmode] gamma;
  vector[nmode] f;
  vector[nmode] phi;
  vector[nsamp] h_det[nobs];

  f[1] = f0;
  gamma[1] = gamma0;

  f[2] = (1+df1)*f_constrained(f0, gamma0);
  gamma[2] = (1+dg1)*g_constrained(f0, gamma0);

  if (gamma[2] < gamma[1]) reject("gamma[2] < gamma[1], so reject");

  for (i in 1:nmode) {
    phi[i] = atan2(xy_phase[i][2], xy_phase[i][1]);
  }

  for (i in 1:nobs) {
    h_det[i] = rep_vector(0.0, nsamp);
    for (j in 1:nmode) {
      h_det[i] = h_det[i] + rd(ts[i]-t0[i], cos_inc, A[j], FpFc[i][1], FpFc[i][2], phi[j], gamma[j], f[j]);
    }
  }
}

model {
  f0 ~ lognormal(mu_logf, sigma_logf);
  gamma0 ~ lognormal(mu_loggamma, sigma_loggamma);

  /* Flat prior on the delta-fs. */

  /* Flat prior on the A. */
  /* Uniform prior on phi. */

  /* Likelihood */
  for (i in 1:nobs) {
    strain[i] ~ multi_normal_cholesky(h_det[i], L[i]);
  }
}
