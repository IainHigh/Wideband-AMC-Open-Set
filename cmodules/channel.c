// 
// awgn.c
//

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <getopt.h>

#include <liquid/liquid.h>
#include "channel.h"
#include "utils.h"

void channel(float snr, int n_sym, int sps, float fo, float po, float xI[], float xQ[],  float yI[], float yQ[],int verbose, int seed)
{

    if (seed < 0){
    	srand((unsigned) time(0));
    }
    else{
	    srand(seed);
    }

    // derived values
    int n_samps = n_sym*sps;
    float nstd = powf(10.0f, -snr/20.0f);

    // init arrays
    float complex x[n_samps];	// transmitted signal
    for (unsigned int i=0; i<n_samps; i++)
    {
	    x[i] = xI[i] + _Complex_I*xQ[i];
    }

    // fo, awgn, sep IQ
    float complex o = cexpf(_Complex_I*(fo + po));
    float complex x_o;
    float complex n;
    float complex y;
    for (unsigned int i=0; i<n_samps; i++)
    {
        // fo & po
        x_o = (crealf(x[i])*crealf(o)) + (crealf(x[i])*cimagf(o)*_Complex_I) + (cimagf(x[i])*crealf(o)*_Complex_I) + (cimagf(x[i])*cimagf(o)*-1.0);

	    // manual awgn
        n = nstd*(randnf() + _Complex_I*randnf())/sqrtf(2.0f);
        y = x_o + n; 

        // split IQ
	    yI[i] = crealf(y);
	    yQ[i] = cimagf(y);	
    }

}

void rayleigh_channel(float snr, int n_sym, int sps, float fo, float po, int num_taps, int awgn, float xI[], float xQ[], float yI[], float yQ[], int verbose, int seed) {
    if (seed < 0) {
        srand((unsigned) time(0));
    } else {
        srand(seed);
    }

    int n_samps = n_sym * sps;
    float nstd = awgn ? powf(10.0f, -snr / 20.0f) : 0.0f;

    // Initialize arrays
    float complex x[n_samps];
    for (unsigned int i = 0; i < n_samps; i++) {
        x[i] = xI[i] + _Complex_I * xQ[i];
    }

    // Generate Rayleigh fading taps
    float complex taps[num_taps];
    for (int i = 0; i < num_taps; i++) {
        float phase = 2.0f * M_PI * ((float) rand() / RAND_MAX);
        taps[i] = cexpf(_Complex_I * phase) / sqrtf(num_taps);
    }

    // Simulate Rayleigh fading
    float complex y[n_samps];
    for (unsigned int i = 0; i < n_samps; i++) {
        y[i] = 0;
        for (int j = 0; j < num_taps; j++) {
            int delay_idx = i - j;
            if (delay_idx >= 0) {
                y[i] += x[delay_idx] * taps[j];
            }
        }

        // Apply frequency offset (fo) and phase offset (po)
        float complex o = cexpf(_Complex_I * (fo * i + po));
        y[i] *= o;

        // Add AWGN if enabled
        if (awgn) {
            float complex noise = nstd * (randnf() + _Complex_I * randnf()) / sqrtf(2.0f);
            y[i] += noise;
        }

        // Separate I/Q components
        yI[i] = crealf(y[i]);
        yQ[i] = cimagf(y[i]);
    }
}

void rician_channel(float snr, int n_sym, int sps, float fo, float po, float k_factor, int num_taps, int awgn, 
                    float xI[], float xQ[], float yI[], float yQ[], float path_delays[], float path_gains[], int verbose, int seed) {
    if (seed < 0) {
        srand((unsigned)time(0));
    } else {
        srand(seed);
    }

    int n_samps = n_sym * sps;
    float nstd = awgn ? powf(10.0f, -snr / 20.0f) : 0.0f;

    // Initialize arrays
    float complex x[n_samps];
    for (unsigned int i = 0; i < n_samps; i++) {
        x[i] = xI[i] + _Complex_I * xQ[i];
    }

    // Generate Rician fading taps
    float complex taps[num_taps];
    float k_linear = powf(10.0f, k_factor / 10.0f); // Convert K-factor to linear scale
    float a = sqrtf(k_linear / (k_linear + 1.0f));  // LOS component amplitude
    float b = sqrtf(1.0f / (k_linear + 1.0f));      // NLOS component amplitude

    for (int i = 0; i < num_taps; i++) {
        float phase = 2.0f * M_PI * ((float)rand() / RAND_MAX);
        taps[i] = a + b * (cexpf(_Complex_I * phase));
    }

    // Simulate Rician fading
    float complex y[n_samps];
    for (unsigned int i = 0; i < n_samps; i++) {
        y[i] = 0;

        for (int j = 0; j < num_taps; j++) {
            int delay_idx = i - (int)(path_delays[j]); // Convert delay to sample offset
            if (delay_idx >= 0) {
                float path_gain = powf(10.0f, path_gains[j] / 20.0f); // Gain for each path
                y[i] += x[delay_idx] * taps[j] * path_gain;          // Apply gain and tap
            }
        }

        // Apply frequency offset (fo) and phase offset (po)
        float complex o = cexpf(_Complex_I * (fo * i + po));
        y[i] *= o;

        // Add AWGN if enabled
        if (awgn) {
            float complex noise = nstd * (randnf() + _Complex_I * randnf()) / sqrtf(2.0f);
            y[i] += noise;
        }

        // Separate I/Q components
        yI[i] = crealf(y[i]);
        yQ[i] = cimagf(y[i]);
    }
}
