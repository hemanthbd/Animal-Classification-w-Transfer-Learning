% Arizona State University
% Ira Fulton School of Engineering
% Electrical Department
% EEE498 - Spring 2005: Real Time DSP


close all;

% Plotting first segment
% Sampling frequency 48Khz
% Target value for 1st segment = 1
% initial Condition = 0
% Attack Time = 30 ms -> 1440 samples

g= 0.0032;
n=0:1439;
%Closed form
y = 1- (1-g).^(n+1);

%Recursive form
g = 0.0032;
yp(1) = 0;
for i=2:1140
yp(i) = .99*g +   (1-g)*yp(i-1)  ;
end

%Plot the rising exponential
figure;
plot(y);
xlabel('Samples');
ylabel('Magnitude');
title('1st Segment');

% Plotting 2nd segment
% Sampling frequency 48Khz
% Target value for 2nd segment = 0.01 
% initial Condition = 0.999
% Sustain Time = (1000 - 30 - 30) ms - > 45120 samples
n=0:45119;
g = 0.00011697;
y1 =.01+0.98*(1-g).^(n+1);
%Recursive form
yp1(1) = 0.99;
for i=2:45120
yp1(i) = 0.01*g +   (1-g)*yp1(i-1)  ;
end
%Plot the decaying exponential

figure
plot(y1);
xlabel('Samples');
ylabel('Magnitude');
title('2nd Segment');

% Plotting 3rd segment
% Sampling frequency 48Khz
% Target value for 2nd segment = 0 
% initial Condition = 0.01
% Release Time = 30 ms - > 1440 samples.

%Closed Form
n=0:1439;
g= 0.0032;
y2 = 0.01*(1-g).^(n+1);

%Recursive form
g = 0.0032;
yp2(1) = 0.01;
for i=2:1140
yp2(i) = (1-g)*yp2(i-1)  ;
end

% Plot the 3rd segment
figure

plot(y2);
xlabel('Samples');
ylabel('Magnitude');
title('3rd Segment');

%Constructing the ADSR envelope
ADSRenv = ([y y1 y2]);


% Generating a one second sine wave 
Fs= 48000;
F = 440;
n=0:1/(Fs-1):1;
z = sin(2*pi*F*n);
% Plotting the modulated signal
figure;
plot(z.*ADSRenv);
title('Plot of the modulated note - 440 Hz Note');
xlabel('Samples');
disp('Press a key to play unmodulated sound')
pause; %wait for a key press

% Testing both umnodulated and modulated sounds

sound(z,Fs)
disp('Press a key to play modulated sound')
pause;
sound(z.*ADSRenv,Fs)

