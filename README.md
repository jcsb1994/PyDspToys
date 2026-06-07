# Python DSP Toys

This is a sandbox for testing ideas with audio DSP.

Objective: (WIP) The goal is to let the user select:
- Input signal (wave type/freq/ampl, or input file from a list of short effects)
- simulated audio chains (or even build their own chain but must think of a clean way to implement through CLI)
- desired analysis (time domain plot, fft, prints of detected harmonics, ...)

Some simulations will not make use of any analysis tool however, so must probably have default analysis for each simulation

## Example chains
```
"sine(3000,0.3)->sine(1000,1)->timeplot"
"sine(1000,1)->bitcrush(32)->fft->timeplot" # Adds little harmonics
"sine(1000,1)->bitcrush(4)->fft" # Adds a lot of harmonics
"sine(3000,0.3)->pan(-100)->sine(1000,0.1)->fft(overlay)" # Pan 2 sines (note: no way to pan each sine individually though)
```