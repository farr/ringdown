functions {
  vector rd(vector t, real Ax, real Ay, real tau, real f) {
    return exp(-t/tau).*(Ax*cos(2*pi()*f*t) + Ay*sin(2*pi()*f*t));
  }
}

data {
  int nobs;
  int nsamp;
  int nmode;

  vector[nsamp] ts[nobs];
  vector[nsamp] strain[nobs];
  matrix[nsamp,nsamp] L[nobs];

  vector[nmode] mu_logf;
  vector[nmode] sigma_logf;

  vector[nmode] mu_logtau;
  vector[nmode] sigma_logtau;

  real Amax;
}

parameters {
  vector<lower=0>[nmode] f;
  vector<lower=0>[nmode] tau;

  vector<lower=0, upper=Amax>[nmode] A[nobs];
  unit_vector[2] xy[nobs, nmode];
}

model {
  f ~ lognormal(mu_logf, sigma_logf);
  tau ~ lognormal(mu_logtau, sigma_logtau);
  /* Flat prior on the A. */
  /* Uniform prior on phi. */

  /* Likelihood */
  for (i in 1:nobs) {
    vector[nsamp] h = rep_vector(0.0, nsamp);
    for (j in 1:nmode) {
      h = h + rd(ts[i], A[i][j]*xy[i,j][1], A[i][j]*xy[i,j][2], tau[j], f[j]);
    }

    strain[i] ~ multi_normal_cholesky(h, L[i]);
  }
}
