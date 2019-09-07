functions {
  vector rd(vector t, real Ax, real Ay, real tau, real f) {
    return exp(-t/tau).*(Ax*cos(2*pi()*f*t) + Ay*sin(2*pi()*f*t));
  }
}

data {
  int nsamp;
  int nmode;

  vector[nsamp] ts;
  vector[nsamp] strain[2];
  matrix[nsamp,nsamp] L[2];

  vector[nmode] mu_logf;
  vector[nmode] sigma_logf;

  vector[nmode] mu_logtau;
  vector[nmode] sigma_logtau;

  real Amax;
}

parameters {
  vector<lower=0>[nmode] f;
  vector<lower=0>[nmode] tau;

  vector<lower=0, upper=Amax>[nmode] A;
  unit_vector[2] xy[nmode];
}

model {
  f ~ lognormal(mu_logf, sigma_logf);
  tau ~ lognormal(mu_logtau, sigma_logtau);
  /* Flat prior on the A. */
  /* Uniform prior on phi. */

  /* Likelihood */
  for (i in 1:2) {
    vector[nsamp] h = rep_vector(0.0, nsamp);

    for (j in 1:nmode) {
      h = h + rd(ts, A[j]*xy[j][1], A[j]*xy[j][2], tau[j], f[j]);
    }

    strain[i] ~ multi_normal_cholesky(h, L[i]);
  }
}
