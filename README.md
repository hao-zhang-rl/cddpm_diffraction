This project is an example for the c-DDPM used to seprate seismic diffractions. We developed the project based on the work of : https://iterative-refinement.github.io/. Note that this project is still under revision.

#Usage

Before starting the test, you must download the trained model from https://drive.google.com/drive/folders/1eMllr4w1s95nhG-p7HAjq7L7x2uS7w0a?usp=drive_link and place the directory "dataset2_consine" under the  directory: ./experiments/

Then, start the diffraction separation by runing 
<pre><code>$mpip install -r requirements.txt</code></pre>

The separate diffraction can be found at ：
./experimets/sr_ffhq_240122_213338/result/

where the SR_1.dat and INF_1.dat are the seprated diffractions and input full-wavefield data.

the size of the dat file is:
nt=625;nx=1387
