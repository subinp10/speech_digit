Speech-Based Digit Recognition System
A speech-based digit recognition system that recognizes spoken digits using Hidden Markov Models (HMMs). The system processes raw speech signals, extracts LPC-based features, converts the continuous feature vectors into discrete observation symbols using LBG Vector Quantization, and performs digit classification using HMM likelihood estimation.
📌 Project Overview
Speech is a time-varying signal, making it difficult to recognize using conventional static classification techniques. This project models speech as a sequence of observations and uses Hidden Markov Models to capture the temporal characteristics of spoken digits.
The system is designed to recognize the digits 0–9 from speech recordings.
The overall pipeline is:
```text
              Speech Signal
                    │
                    ▼
            DC Shift Correction
                    │
                    ▼
              Normalization
                    │
                    ▼
             Frame Extraction
                    │
                    ▼
             Steady-State Frames
                    │
                    ▼
              LPC Analysis
                    │
                    ▼
          Cepstral Coefficients
                    │
                    ▼
        LBG Vector Quantization
                    │
                    ▼
        Observation Symbol Sequence
                    │
                    ▼
            HMM Training/Testing
                    │
                    ▼
          Recognized Digit (0–9)
```
---
✨ Features
Spoken digit recognition using Hidden Markov Models
Speech preprocessing and normalization
Frame-based speech analysis
LPC-based feature extraction
Cepstral coefficient computation
LBG vector quantization
Discrete observation generation
Forward algorithm for HMM evaluation
Baum-Welch algorithm for HMM training
Support for multiple HMM configurations
C/C++ implementation
---
🧠 Algorithms Used
1. Linear Predictive Coding (LPC)
LPC models a speech signal by predicting the current sample from previous samples.
The predicted sample is represented as:
[
\hat{s}(n)=\sum_{k=1}^{p}a_k s(n-k)
]
where:
(s(n)) = current speech sample
(a_k) = LPC coefficients
(p) = LPC order
The prediction error is:
[
e(n)=s(n)-\hat{s}(n)
]
In this project, LPC coefficients are used to characterize the spectral properties of speech.
---
2. Cepstral Coefficients
The LPC coefficients are converted into cepstral coefficients to obtain a more useful representation of the speech signal.
The resulting feature vector represents the characteristics of each speech frame.
These features are later passed to the vector quantization stage.
---
3. LBG Vector Quantization
The speech feature vectors are continuous-valued, whereas the discrete HMM requires observation symbols.
Linde-Buzo-Gray (LBG) vector quantization is used to construct a codebook.
```text
Continuous Feature Vector
          │
          ▼
     LBG Algorithm
          │
          ▼
       Codebook
          │
          ▼
Nearest Codeword
          │
          ▼
Observation Symbol
```
Each speech feature vector is mapped to the closest codeword using a distance measure.
The resulting sequence might look like:
```text
12  7  7  15  21  21  18  9  9  14
```
These values become the observations for the HMM.
---
🔬 Hidden Markov Model
Each digit is represented by an individual HMM.
For example:
```text
Digit 0 → HMM₀
Digit 1 → HMM₁
Digit 2 → HMM₂
...
Digit 9 → HMM₉
```
An HMM is defined by:
[
\lambda=(A,B,\pi)
]
where:
(A) = state transition probability matrix
(B) = observation probability matrix
(\pi) = initial state probability distribution
For example, an HMM may contain:
```text
State 1 → State 2 → State 3 → State 4 → State 5
```
Each state represents a different portion of the speech sequence.
---
🎯 HMM Evaluation
During testing, the system receives an observation sequence:
[
O=(o_1,o_2,\ldots,o_T)
]
The evaluation problem is to calculate:
[
P(O|\lambda)
]
This is performed using the Forward Algorithm.
The forward variable is:
[
\alpha_t(i)=P(o_1,o_2,\ldots,o_t,q_t=i|\lambda)
]
Initialization
[
\alpha_1(i)=\pi_i b_i(o_1)
]
Induction
[
\alpha_t(j)=
\left[
\sum_{i=1}^{N}
\alpha_{t-1}(i)a_{ij}
\right]
b_j(o_t)
]
Termination
[
P(O|\lambda)=\sum_{i=1}^{N}\alpha_T(i)
]
The digit model producing the highest likelihood is selected.
---
🔄 HMM Training
The HMM parameters are estimated using the Baum-Welch algorithm, which is an Expectation-Maximization (EM) algorithm.
The training process repeatedly:
Computes forward probabilities.
Computes backward probabilities.
Estimates state occupation probabilities.
Estimates state transition probabilities.
Updates observation probabilities.
Repeats until convergence.
The goal is to maximize:
[
P(O|\lambda)
]
for the training speech samples.
---
📊 Forward Algorithm
The Forward Algorithm efficiently evaluates the probability of an observation sequence.
Without dynamic programming, all possible state sequences would need to be considered:
[
N^T
]
possible state sequences.
The Forward Algorithm reduces this to:
[
O(N^2T)
]
where:
(N) = number of HMM states
(T) = number of observations
---
🔁 Baum-Welch Algorithm
The Baum-Welch algorithm updates the HMM parameters using the training observations.
The main quantities are:
State probability
[
\gamma_t(i)
P(q_t=i|O,\lambda)
]
Transition probability
[
\xi_t(i,j)
P(q_t=i,q_{t+1}=j|O,\lambda)
]
The parameters are then re-estimated from these probabilities.
---
🛠️ Speech Processing Pipeline
1. DC Shift Correction
Removes the DC component from the recorded speech signal.
2. Normalization
The amplitude of the signal is normalized to reduce variations caused by recording conditions.
3. Frame Extraction
Speech is divided into short frames because speech characteristics can be considered approximately stationary over a short period.
Example parameters:
```text
Frame Size  : 320 samples
Frame Shift : 10 ms
```
4. Steady-State Frame Selection
The stable portion of the spoken digit is selected for feature extraction.
5. LPC Analysis
LPC coefficients are calculated for each selected frame.
Example:
```text
LPC Order = 12
```
6. Cepstral Analysis
LPC coefficients are transformed into cepstral coefficients.
7. Vector Quantization
Each feature vector is mapped to its nearest codeword in the LBG codebook.
8. HMM Processing
The resulting observation sequence is used to train or evaluate the digit HMM.
---
🚀 Training
The training phase creates a separate HMM for each digit.
```text
Training Speech
      │
      ▼
Preprocessing
      │
      ▼
Feature Extraction
      │
      ▼
Vector Quantization
      │
      ▼
Observation Sequences
      │
      ▼
Baum-Welch Training
      │
      ▼
Digit HMM Models

💻 Technologies
C/C++
Hidden Markov Models
Linear Predictive Coding
Digital Signal Processing
Vector Quantization
LBG Algorithm
Probability and Statistical Modeling
---
📚 Concepts Demonstrated
This project demonstrates practical implementation of:
Digital Signal Processing
Speech Signal Processing
Feature Extraction
Linear Predictive Coding
Cepstral Analysis
Vector Quantization
Hidden Markov Models
Dynamic Programming
Forward Algorithm
Baum-Welch Algorithm
Probability Modeling
Sequence Classification
---
🎓 Learning Outcomes
Through this project, the following concepts are applied in an end-to-end system:
```text
Raw Signal
    ↓
Signal Processing
    ↓
Feature Representation
    ↓
Vector Quantization
    ↓
Probabilistic Modeling
    ↓
Sequence Classification
```
The project provides an understanding of how traditional speech recognition systems can be constructed without relying on modern deep-learning frameworks.
---
🔗 Repository
GitHub:
https://github.com/subinp10/speech_digit
---
👨‍💻 Author
Subin Pullambalavan
M.Tech Computer Science
This project was developed as a speech signal processing and pattern recognition application using classical statistical and signal-processing techniques.
