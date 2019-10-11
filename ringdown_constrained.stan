functions {
  real f0_factor(real chi) {
    real log1mc = log1p(-chi);

    return -0.00823557*log1mc + 0.05994978 + chi*(-0.00106621 + chi*(0.08354181 + chi*(-0.15165638 + chi*0.11021346)));
  }

  real f1_factor(real chi) {
    real log1mc = log1p(-chi);

    return -0.00817462*log1mc + 0.05566214 + chi*(0.00174686 + chi*(0.08531498 + chi*(-0.15464552 + chi*0.11325923)));
  }

  real g0_factor(real chi) {
    real log1mc = log1p(-chi);

    return 0.01180702*log1mc + 0.08838127 + chi*(0.02528302 + chi*(-0.09002286 + chi*(0.18245511 - chi*0.12162592)));
  }

  real g1_factor(real chi) {
    real log1mc = log1p(-chi);

    return 0.03544845*log1mc + 0.27212686 + chi*(0.0702625 + chi*(-0.27797961 + chi*(0.55851284 - chi*0.36945098)));
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

  /* Priors on m and chi are flat */
  real MMin;
  real MMax;

  vector[2] FpFc[nobs];

  real cos_inc;

  real Amax;

  real df_max;
  real dtau_max;
}

transformed data {
  int nmode = 2;
}

parameters {
  real<lower=MMin, upper=MMax> M;
  real<lower=0, upper=1> chi;

  real<lower=-df_max, upper=df_max> df1;
  real<lower=-dtau_max, upper=dtau_max> dtau1;

  vector<lower=-Amax, upper=Amax>[nmode] Ax;
  vector<lower=-Amax, upper=Amax>[nmode] Ay;
}

transformed parameters {
  vector[nmode] gamma;
  vector[nmode] f;
  vector[nmode] phi;
  vector[nmode] A = sqrt(Ax .* Ax + Ay .* Ay);
  vector[nsamp] h_det[nobs];

  for (i in 1:nmode) {
    phi[i] = atan2(Ay[i], Ax[i]);
  }

  {
    real fref = 2980.0;
    real mref = 68.0;

    real f0 = fref*mref/M;

    f[1] = f0*f0_factor(chi);
    f[2] = f0*f1_factor(chi)*(1 + df1);
    gamma[1] = f0*g0_factor(chi);
    gamma[2] = f0*g1_factor(chi)/(1 + dtau1);
  }

  if (gamma[2] < gamma[1]) reject("gamma[2] < gamma[1], so reject");

  for (i in 1:nobs) {
    h_det[i] = rep_vector(0.0, nsamp);
    for (j in 1:nmode) {
      h_det[i] = h_det[i] + rd(ts[i]-t0[i], cos_inc, A[j], FpFc[i][1], FpFc[i][2], phi[j], gamma[j], f[j]);
    }
  }
}

model {
  /* Flat prior on M, chi */

  /* Flat prior on the delta-fs. */

  /* Flat prior on the |A| => 1/|A| jacobian to Ax, Ay. */
  target += -sum(log(A));

  /* Uniform prior on phi. */

  /* Likelihood */
  for (i in 1:nobs) {
    strain[i] ~ multi_normal_cholesky(h_det[i], L[i]);
  }
}
