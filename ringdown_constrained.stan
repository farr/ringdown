functions {
  real f_constrained(real f0, real gamma0) {
    real x = log(f0/gamma0);

    real log1mf = -3.71411153 + x*(-2.76591614 + x*(-0.23758466 + x*(0.15179355 - x*0.03618724)));

    return -f0*expm1(log1mf);
  }

  real g_constrained(real f0, real gamma0) {
    real x = log(f0/gamma0);

    real loggamm3 = -3.59764966 + x*(-2.30739315 + x*(0.42826231 + x*(-0.96875461 + x*0.30154111)));

    return gamma0*(3 + exp(loggamm3));
  }

  real chi(real f0, real gamma0) {
    real x = log(f0/gamma0);

    real log1mchi = -1.0804979 + x*(-2.5688046 + x*(0.27351776 + x*(-0.05828812 + x*0.00361861)));

    return -expm1(log1mchi);
  }

  real ffactor(real f0, real gamma0) {
    real x = log(f0/gamma0);

    return 0.08215104 + x*(0.0542529 + x*(-0.00804415 + x*(-0.00415015 + x*0.00123021)));
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
  real<lower=exp(-2.670)*f0, upper=exp(0.403)*f0> gamma0;

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

generated quantities {
  real M0 = 68.0*2.98e3/(f0/ffactor(f0,gamma0));
  real chi0 = chi(f0, gamma0);
}
