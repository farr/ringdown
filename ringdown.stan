functions {
  vector rd(vector t, real Ax, real Ay, real gamma, real f) {
    return exp(-t*gamma).*(Ax*cos(2*pi()*f*t) + Ay*sin(2*pi()*f*t));
  }
}

data {
  int nobs;
  int nsamp;
  int nmode;

  real t0[nobs];
  vector[nsamp] ts[nobs];
  vector[nsamp] strain[nobs];
  matrix[nsamp,nsamp] L[nobs];

  vector[nmode] mu_logf;
  vector[nmode] sigma_logf;

  vector[nmode] mu_loggamma;
  vector[nmode] sigma_loggamma;

  vector[2] FpFc[nobs];

  real Amax;
}

parameters {
  vector<lower=0>[nmode] f;
  positive_ordered[nmode] gamma;

  vector<lower=0, upper=Amax>[nmode] A;
  unit_vector[2] xy[nmode];
}

transformed parameters {
  vector[nsamp] h_det[nobs];

  for (i in 1:nobs) {
    h_det[i] = rep_vector(0.0, nsamp);
    for (j in 1:nmode) {
      h_det[i] = h_det[i] + rd(ts[i]-t0[i], FpFc[i][1]*A[j]*xy[j][1], FpFc[i][2]*A[j]*xy[j][2], gamma[j], f[j]);
    }
  }
}

model {
  f ~ lognormal(mu_logf, sigma_logf);
  gamma ~ lognormal(mu_loggamma, sigma_loggamma);
  /* Flat prior on the A. */
  /* Uniform prior on phi. */

  /* Likelihood */
  for (i in 1:nobs) {
    strain[i] ~ multi_normal_cholesky(h_det[i], L[i]);
  }
}
