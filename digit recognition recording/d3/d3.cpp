

#define _CRT_SECURE_NO_WARNINGS
#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")

using namespace std;


#define FRAME_LEN_MS 20      
#define FRAME_SHIFT_MS 10     
#define SAMPLE_RATE 16000
#define FRAME_LEN (SAMPLE_RATE * FRAME_LEN_MS / 1000)
#define FRAME_SHIFT (SAMPLE_RATE * FRAME_SHIFT_MS / 1000)

#define STATES 8
#define SYMBOLS 64
#define MAX_FRAMES 300
#define MAX_SEQ_LEN 300
#define MAX_SAMPLES 48000
#define EM_ITERS 12
#define KMEANS_ITERS 20
#define RANDOM_SEED 12345

#define RECORD_SECONDS 2
#define RECORD_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)
//mention directory of the files
string DATA_FOLDER = "C:\\Users\\SUBIN P\\Desktop\\speech\\New folder (2)\\254101058_datsaset\\English\\txt\\";
string MODELS_DIR = "C:\\Users\\SUBIN P\\Desktop\\speech\\New folder (2)\\254101058_datsaset\\English\\models\\";

/* ------------------------ HMM struct ------------------------ */
struct HMM {
    int N, M;
    double pi[STATES];
    double A[STATES][STATES];
    double B[STATES][SYMBOLS];
    HMM() { 
        N=STATES; M=SYMBOLS; 
        for(int i=0;i<STATES;i++){
            pi[i]=0; 
            for(int j=0;j<STATES;j++) A[i][j]=0; 
            for(int k=0;k<SYMBOLS;k++) B[i][k]=0;
        } 
    }
};

/* ------------------------ Global arrays (reduced size) ------------------------ */
double alpha[MAX_SEQ_LEN][STATES];
double beta_mat[MAX_SEQ_LEN][STATES];
double gamma_mat[MAX_SEQ_LEN][STATES];
double xi_mat[MAX_SEQ_LEN][STATES][STATES];
double scale_arr[MAX_SEQ_LEN];

/* ------------------------ Utilities ------------------------ */
inline void normalize_arr(double *arr, int len) {
    double s = 0;
    for (int i=0;i<len;i++) s += arr[i];
    if (s <= 0) { 
        double v = 1.0 / len; 
        for (int i=0;i<len;i++) arr[i] = v; 
        return; 
    }
    for (int i=0;i<len;i++) arr[i] /= s;
}

void init_hmm_left_right(HMM &h) {
    int N = h.N;
    for (int i=0;i<N;i++) 
        for (int j=0;j<N;j++) 
            h.A[i][j] = 0.0;
    
    for (int i=0;i<N;i++) {
        if (i == N-1) 
            h.A[i][i] = 1.0;
        else { 
            h.A[i][i] = 0.6; 
            h.A[i][i+1] = 0.4; 
        }
    }
    
    for (int i=0;i<N;i++) 
        for (int k=0;k<h.M;k++) 
            h.B[i][k] = 1.0 / (double)h.M;
    
    for (int i=0;i<N;i++) 
        h.pi[i] = (i==0?1.0:0.0);
}

/* ------------------------ Feature extraction ------------------------ */
void compute_frames_ste_zcr(const double *samples, int nsamples, vector< vector<double> > &frames) {
    frames.clear();
    if (nsamples <= FRAME_LEN) return;
    
    int pos = 0;
    while (pos + FRAME_LEN <= nsamples) {
        double ste = 0.0;
        int zcr = 0;
        double prev = samples[pos];
        
        for (int i=0;i<FRAME_LEN;i++) {
            double s = samples[pos + i];
            ste += s * s;
            if (i>0) {
                if ((s>=0 && prev<0) || (s<0 && prev>=0)) 
                    zcr++;
            }
            prev = s;
        }
        
        vector<double> feat(2);
        feat[0] = log(1.0 + ste);
        feat[1] = (double)zcr / FRAME_LEN;
        frames.push_back(feat);
        
        pos += FRAME_SHIFT;
        if ((int)frames.size() >= MAX_FRAMES) break;
    }
}

/* ------------------------ Normalization ------------------------ */
void normalize_feature_vectors(vector< vector<double> > &all_feats, double *mean_out, double *std_out) {
    int dim=2; 
    for(int d=0;d<dim;d++) 
        mean_out[d]=std_out[d]=0.0;
    
    long count=0;
    for(size_t i=0;i<all_feats.size();i++){
        for(int d=0;d<dim;d++) 
            mean_out[d]+=all_feats[i][d]; 
        count++;
    }
    
    if(count==0){
        for(int d=0;d<dim;d++){
            mean_out[d]=0;
            std_out[d]=1;
        } 
        return;
    }
    
    for(int d=0;d<dim;d++) 
        mean_out[d]/=(double)count;
    
    for(size_t i=0;i<all_feats.size();i++){
        for(int d=0;d<dim;d++){
            double diff=all_feats[i][d]-mean_out[d]; 
            std_out[d]+=diff*diff;
        }
    }
    
    for(int d=0;d<dim;d++){
        std_out[d]=sqrt(std_out[d]/(double)count); 
        if(std_out[d]<=1e-6) std_out[d]=1.0;
    }
    
    for(size_t i=0;i<all_feats.size();i++) 
        for(int d=0;d<dim;d++) 
            all_feats[i][d]=(all_feats[i][d]-mean_out[d])/std_out[d];
}

/* ------------------------ LBG K-means ------------------------ */
double euclid_dist2(const double *a, const double *b, int dim) { 
    double s=0; 
    for(int i=0;i<dim;i++){
        double v=a[i]-b[i]; 
        s+=v*v;
    } 
    return s; 
}

void lbg_kmeans(const vector< vector<double> > &data, int codebook_size, vector< vector<double> > &centroids) {
    int dim=2; 
    centroids.clear(); 
    int N=(int)data.size();
    
    for(int k=0;k<codebook_size;k++){
        int idx=(k*N)/codebook_size; 
        vector<double> c(dim); 
        for(int d=0;d<dim;d++) 
            c[d]=data[idx][d]; 
        centroids.push_back(c);
    }
    
    int maxiter=KMEANS_ITERS; 
    vector<int> assign(N,0);
    
    for(int it=0;it<maxiter;it++){
        bool changed=false;
        
        // Assignment step
        for(int i=0;i<N;i++){
            double best=1e300; 
            int bestk=0;
            for(int k=0;k<codebook_size;k++){
                double s=0; 
                for(int d=0;d<dim;d++){
                    double diff=data[i][d]-centroids[k][d]; 
                    s+=diff*diff;
                } 
                if(s<best){
                    best=s; 
                    bestk=k;
                }
            }
            if(assign[i]!=bestk){
                assign[i]=bestk; 
                changed=true;
            }
        }
        
        // Update step
        vector< vector<double> > sum(codebook_size, vector<double>(dim,0.0)); 
        vector<int> cnt(codebook_size,0);
        for(int i=0;i<N;i++){
            int k=assign[i]; 
            cnt[k]++; 
            for(int d=0;d<dim;d++) 
                sum[k][d]+=data[i][d];
        }
        
        for(int k=0;k<codebook_size;k++){
            if(cnt[k]>0){
                for(int d=0;d<dim;d++) 
                    centroids[k][d]=sum[k][d]/cnt[k];
            } else {
                int idx=rand()%N; 
                for(int d=0;d<dim;d++) 
                    centroids[k][d]=data[idx][d];
            }
        }
        
        if(!changed) break;
    }
}

int quantize_vector_to_symbol(const vector<double> &vec, const vector< vector<double> > &centroids) {
    int bestk=0; 
    double best=1e300; 
    int dim=(int)vec.size();
    for(size_t k=0;k<centroids.size();k++){
        double s=0; 
        for(int d=0;d<dim;d++){
            double diff=vec[d]-centroids[k][d]; 
            s+=diff*diff;
        } 
        if(s<best){
            best=s; 
            bestk=(int)k;
        }
    }
    return bestk;
}

/* ------------------------ HMM (Forward + Baum-Welch) ------------------------ */
double forward_scaled_loglik(const HMM &hmm, const int *O, int T) {
    if(T<=0) return -1e300; 
    int N=hmm.N;
    
    // Initialize alpha[0][i]
    double c0=0; 
    for(int i=0;i<N;i++){
        alpha[0][i]=hmm.pi[i]*hmm.B[i][O[0]]; 
        c0+=alpha[0][i];
    } 
    if(c0<=0) c0=1e-300; 
    scale_arr[0]=c0; 
    for(int i=0;i<N;i++) 
        alpha[0][i]/=c0;
    
    // Forward pass
    for(int t=1;t<T;t++){
        double ct=0; 
        int ot=O[t]; 
        for(int j=0;j<N;j++){
            double s=0; 
            for(int i=0;i<N;i++) 
                s+=alpha[t-1][i]*hmm.A[i][j]; 
            alpha[t][j]=s*hmm.B[j][ot]; 
            ct+=alpha[t][j];
        } 
        if(ct<=0) ct=1e-300; 
        scale_arr[t]=ct; 
        for(int j=0;j<N;j++) 
            alpha[t][j]/=ct;
    }
    
    double loglik=0; 
    for(int t=0;t<T;t++) 
        loglik+=log(scale_arr[t]); 
    return -loglik;
}

void baum_welch_train(HMM &hmm, int **seqs, int *seq_lens, int S) {
    int N=hmm.N, M=hmm.M;
    if(S<=0) return;
    
    for(int iter=0;iter<EM_ITERS;iter++){
        double A_num[STATES][STATES] = {0}; 
        double A_den[STATES] = {0}; 
        double B_num[STATES][SYMBOLS] = {0}; 
        double B_den[STATES] = {0}; 
        double pi_acc[STATES] = {0};
        
        for(int s=0;s<S;s++){
            int *O=seqs[s]; 
            int T=seq_lens[s]; 
            if(T<=0) continue;
            
            // Forward algorithm with scaling
            double c0=0; 
            for(int i=0;i<N;i++){
                alpha[0][i]=hmm.pi[i]*hmm.B[i][O[0]]; 
                c0+=alpha[0][i];
            } 
            if(c0<=0) c0=1e-300; 
            scale_arr[0]=c0; 
            for(int i=0;i<N;i++) 
                alpha[0][i]/=c0;
            
            for(int t=1;t<T;t++){
                double ct=0; 
                int ot=O[t]; 
                for(int j=0;j<N;j++){
                    double ssum=0; 
                    for(int i=0;i<N;i++) 
                        ssum+=alpha[t-1][i]*hmm.A[i][j]; 
                    alpha[t][j]=ssum*hmm.B[j][ot]; 
                    ct+=alpha[t][j];
                } 
                if(ct<=0) ct=1e-300; 
                scale_arr[t]=ct; 
                for(int j=0;j<N;j++) 
                    alpha[t][j]/=ct;
            }
            
            // Backward algorithm
            for(int i=0;i<N;i++) 
                beta_mat[T-1][i]=1.0/scale_arr[T-1];
            
            for(int t=T-2;t>=0;t--){
                int ot1=O[t+1]; 
                for(int i=0;i<N;i++){
                    double ssum=0; 
                    for(int j=0;j<N;j++) 
                        ssum+=hmm.A[i][j]*hmm.B[j][ot1]*beta_mat[t+1][j]; 
                    beta_mat[t][i]=ssum/scale_arr[t];
                }
            }
            
            // Compute gamma and xi
            for(int t=0;t<T-1;t++){
                double denom=0; 
                int ot1=O[t+1]; 
                for(int i=0;i<N;i++) 
                    for(int j=0;j<N;j++) 
                        denom+=alpha[t][i]*hmm.A[i][j]*hmm.B[j][ot1]*beta_mat[t+1][j]; 
                
                if(denom<=0) denom=1e-300; 
                
                for(int i=0;i<N;i++){
                    gamma_mat[t][i]=0; 
                    for(int j=0;j<N;j++){
                        double val=(alpha[t][i]*hmm.A[i][j]*hmm.B[j][ot1]*beta_mat[t+1][j])/denom; 
                        xi_mat[t][i][j]=val; 
                        gamma_mat[t][i]+=val; 
                        A_num[i][j]+=val; 
                        A_den[i]+=val;
                    }
                }
            }
            
            // Final gamma values
            double denomLast=0; 
            for(int i=0;i<N;i++) 
                denomLast+=alpha[T-1][i]; 
            if(denomLast<=0) denomLast=1e-300; 
            for(int i=0;i<N;i++) 
                gamma_mat[T-1][i]=alpha[T-1][i]/denomLast;
            
            // Accumulate initial state probabilities
            for(int i=0;i<N;i++) 
                pi_acc[i]+=gamma_mat[0][i];
            
            // Accumulate observation probabilities
            for(int t=0;t<T;t++){
                int ot=O[t]; 
                for(int i=0;i<N;i++){
                    B_num[i][ot]+=gamma_mat[t][i]; 
                    B_den[i]+=gamma_mat[t][i];
                }
            }
        }
        
        // Re-estimate parameters
        for(int i=0;i<N;i++){
            hmm.pi[i]=pi_acc[i]/(double)S; 
            if(hmm.pi[i]<1e-12) hmm.pi[i]=1e-12;
        } 
        normalize_arr(hmm.pi,N);
        
        for(int i=0;i<N;i++){
            double denom = A_den[i];
            if(denom<=0) denom=1e-300; 
            for(int j=0;j<N;j++){
                hmm.A[i][j]=A_num[i][j]/denom; 
                if(hmm.A[i][j]<1e-12) hmm.A[i][j]=1e-12;
            } 
            normalize_arr(hmm.A[i],N);
        }
        
        for(int i=0;i<N;i++){
            double denom=B_den[i]; 
            if(denom<=0) denom=1e-300; 
            for(int k=0;k<M;k++){
                hmm.B[i][k]=B_num[i][k]/denom; 
                if(hmm.B[i][k]<1e-12) hmm.B[i][k]=1e-12;
            } 
            normalize_arr(hmm.B[i],M);
        }
    }
}

/* ------------------------ Load/save codebook ------------------------ */
bool save_codebook(const vector< vector<double> > &centroids, const string &path) {
    FILE *f=fopen(path.c_str(),"wt"); 
    if(!f) return false;
    
    int K=(int)centroids.size(); 
    int dim=(K>0)?(int)centroids[0].size():0;
    fprintf(f,"%d %d\n",K,dim);
    
    for(int k=0;k<K;k++){
        for(int d=0;d<dim;d++){
            if(d) fprintf(f," "); 
            fprintf(f,"%.12g",centroids[k][d]);
        } 
        fprintf(f,"\n");
    }
    
    fclose(f); 
    return true;
}

/* ------------------------  recording ------------------------ */
bool record_audio(double *out, int &out_len) {
    out_len=0;
    HWAVEIN hWaveIn;
    WAVEFORMATEX wf;
    wf.wFormatTag = WAVE_FORMAT_PCM;
    wf.nChannels = 1;
    wf.nSamplesPerSec = SAMPLE_RATE;
    wf.nAvgBytesPerSec = SAMPLE_RATE * 2;
    wf.nBlockAlign = 2;
    wf.wBitsPerSample = 16;
    wf.cbSize = 0;
    
    if(waveInOpen(&hWaveIn, WAVE_MAPPER, &wf, 0, 0, CALLBACK_NULL) != MMSYSERR_NOERROR){
        cout<<"Error: cannot open microphone.\n"; 
        return false;
    }
    
    short *buffer = new short[RECORD_SAMPLES];
    WAVEHDR header; 
    ZeroMemory(&header, sizeof(header));
    header.lpData = (LPSTR)buffer; 
    header.dwBufferLength = RECORD_SAMPLES * sizeof(short);
    
    if(waveInPrepareHeader(hWaveIn, &header, sizeof(header)) != MMSYSERR_NOERROR){
        cout<<"Error: cannot prepare header.\n";
        delete[] buffer;
        waveInClose(hWaveIn);
        return false;
    }
    
    if(waveInAddBuffer(hWaveIn, &header, sizeof(header)) != MMSYSERR_NOERROR){
        cout<<"Error: cannot add buffer.\n";
        delete[] buffer;
        waveInClose(hWaveIn);
        return false;
    }
    
    if(waveInStart(hWaveIn) != MMSYSERR_NOERROR){
        cout<<"Error: cannot start recording.\n";
        delete[] buffer;
        waveInClose(hWaveIn);
        return false;
    }
    
    cout<<"Recording for "<<RECORD_SECONDS<<" seconds...\n";
    Sleep(RECORD_SECONDS * 1000);
    
    waveInStop(hWaveIn);
    waveInUnprepareHeader(hWaveIn, &header, sizeof(header));
    waveInClose(hWaveIn);
    
    out_len = RECORD_SAMPLES;
    for(int i=0;i<RECORD_SAMPLES;i++) 
        out[i] = buffer[i] / 32768.0;
    
    delete[] buffer;
    return true;
}

/* ------------------------ MAIN ------------------------ */
int main() {
    srand(RANDOM_SEED);
    cout << "DATA_FOLDER = " << DATA_FOLDER << "\n";
    cout << "MODELS_DIR = " << MODELS_DIR << "\n";

    // Load precomputed codebook
    vector< vector<double> > centroids;
    char codepath[1024]; 
    sprintf(codepath, "%scodebook.txt", MODELS_DIR.c_str());
    ifstream fcode(codepath);
    
    if(!fcode.is_open()){
        cout<<"Cannot open codebook file: "<<codepath<<"\n"; 
        return 1;
    }
    
    int K, dim; 
    fcode >> K >> dim; 
    centroids.resize(K, vector<double>(dim, 0.0));
    
    for(int k=0;k<K;k++) 
        for(int d=0;d<dim;d++) 
            fcode >> centroids[k][d];
    
    fcode.close();
    cout << "Loaded codebook with " << K << " centroids, dimension " << dim << "\n";

    // Load trained HMM models
    HMM models[10];
    bool models_loaded = true;
    
    for(int d=0;d<10;d++){
        char mpath[1024]; 
        sprintf(mpath, "%sdigit_%d.txt", MODELS_DIR.c_str(), d);
        ifstream fm(mpath); 
        
        if(!fm.is_open()){
            cout<<"Cannot open model file: "<<mpath<<"\n"; 
            models_loaded = false;
            break;
        }
        
        fm >> models[d].N >> models[d].M;
        for(int i=0;i<models[d].N;i++) 
            fm >> models[d].pi[i];
        for(int i=0;i<models[d].N;i++) 
            for(int j=0;j<models[d].N;j++) 
                fm >> models[d].A[i][j];
        for(int i=0;i<models[d].N;i++) 
            for(int k=0;k<models[d].M;k++) 
                fm >> models[d].B[i][k];
        
        fm.close();
    }
    
    if(!models_loaded){
        cout << "Failed to load HMM models.\n";
        return 1;
    }
    
    cout << "Loaded all HMM models successfully.\n";

    // Use default normalization (you may need to load precomputed values)
    double meanv[2] = {0.0, 0.0};
    double stdv[2] = {1.0, 1.0};

    // RECORD LIVE AUDIO
    cout << "\nStarting audio recording...\n";
    double *live_samples = new double[MAX_SAMPLES];
    int nsamp = 0;
    
    if(!record_audio(live_samples, nsamp)){
        cout << "Recording failed.\n";
        delete[] live_samples;
        return 1;
    }
    
    cout << "Recorded " << nsamp << " samples.\n";

    // Extract features
    vector< vector<double> > frames;
    compute_frames_ste_zcr(live_samples, nsamp, frames);
    
    if(frames.empty()){
        cout << "No frames extracted from audio.\n";
        delete[] live_samples;
        return 1;
    }
    
    cout << "Extracted " << frames.size() << " frames.\n";

    // Normalize features
    for(size_t i=0;i<frames.size();i++) 
        for(int k=0;k<2;k++) 
            frames[i][k] = (frames[i][k] - meanv[k]) / stdv[k];

    // Quantize to symbols
    int T = (int)frames.size();
    int *obs = new int[T];
    
    for(int i=0;i<T;i++) 
        obs[i] = quantize_vector_to_symbol(frames[i], centroids);

    // Run HMM forward to predict
    double bestScore = -1e300; 
    int bestLabel = -1;
    
    for(int m=0;m<10;m++){
        double neglog = forward_scaled_loglik(models[m], obs, T); 
        double lik = -neglog; 
        if(lik > bestScore){
            bestScore = lik; 
            bestLabel = m;
        }
        cout << "Digit " << m << " score: " << lik << "\n";
    }
    
    printf("\nPredicted digit = %d  Score = %.6g  Frames = %d\n", bestLabel, bestScore, T);

    // Cleanup
    delete[] obs;
    delete[] live_samples;
    
    cout << "Press ENTER to exit...\n"; 
    getchar();
    
    return 0;
}