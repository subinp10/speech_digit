
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

#ifdef _WIN32
#include <direct.h> 
#endif

using namespace std;


#define FRAME_LEN_MS 20       
#define FRAME_SHIFT_MS 10     
#define SAMPLE_RATE 16000     
#define FRAME_LEN (SAMPLE_RATE * FRAME_LEN_MS / 1000)
#define FRAME_SHIFT (SAMPLE_RATE * FRAME_SHIFT_MS / 1000)

#define STATES 8
#define SYMBOLS 64            
#define MAX_FRAMES 2000
#define MAX_SEQ_LEN 2000
#define MAX_SAMPLES 200000
#define EM_ITERS 12
#define KMEANS_ITERS 20
#define RANDOM_SEED 12345

//mention directory of the files
string DATA_FOLDER = "C:\\Users\\SUBIN P\\Desktop\\speech\\New folder (2)\\254101058_datsaset\\English\\txt\\";
string MODELS_DIR = "C:\\Users\\SUBIN P\\Desktop\\speech\\New folder (2)\\254101058_datsaset\\English\\models\\"; 
/* ------------------------ HMM struct ------------------------ */
struct HMM {
    int N, M;
    double pi[STATES];
    double A[STATES][STATES];
    double B[STATES][SYMBOLS];
    HMM() { N=STATES; M=SYMBOLS; for(int i=0;i<STATES;i++){pi[i]=0; for(int j=0;j<STATES;j++)A[i][j]=0; for(int k=0;k<SYMBOLS;k++)B[i][k]=0;} }
};

/* ------------------------ Utilities ------------------------ */
inline void normalize_arr(double *arr, int len) {
    double s = 0;
    for (int i=0;i<len;i++) s += arr[i];
    if (s <= 0) { double v = 1.0 / len; for (int i=0;i<len;i++) arr[i] = v; return; }
    for (int i=0;i<len;i++) arr[i] /= s;
}

void init_hmm_left_right(HMM &h) {
    int N = h.N;
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) h.A[i][j] = 0.0;
    for (int i=0;i<N;i++) {
        if (i == N-1) h.A[i][i] = 1.0;
        else { h.A[i][i] = 0.6; h.A[i][i+1] = 0.4; }
    }
    for (int i=0;i<N;i++) for (int k=0;k<h.M;k++) h.B[i][k] = 1.0 / (double)h.M;
    for (int i=0;i<N;i++) h.pi[i] = (i==0?1.0:0.0);
}

/* ------------------------ File loader (skips first 9 header lines) ------------------------ */
// loads samples as doubles into out[], returns count in out_len (0 on error)
int load_samples_txt(const string &path, double *out, int &out_len) {
    out_len = 0;
    ifstream fin(path.c_str());
    if (!fin.is_open()) return 0;
    string line;
    // skip first 9 lines (assumed header)
    for (int i=0;i<9;i++) {
        if (!getline(fin, line)) { fin.close(); return 0; }
    }
    while (getline(fin, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        double v;
        while (ss >> v) {
            if (out_len >= MAX_SAMPLES) break;
            out[out_len++] = v;
        }
        if (out_len >= MAX_SAMPLES) break;
    }
    fin.close();
    return out_len;
}

/* ------------------------ Feature extraction: frame-level STE & ZCR ------------------------ */
// compute per-frame STE and ZCR, store as 2D vector (frame_count x 2)
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
            if (i > 0) {
                if ((s >= 0 && prev < 0) || (s < 0 && prev >= 0)) zcr++;
            }
            prev = s;
        }
        vector<double> feat(2);
        feat[0] = log(1.0 + ste); // log-energy
        feat[1] = (double)zcr / FRAME_LEN; // normalized zcr
        frames.push_back(feat);
        pos += FRAME_SHIFT;
        if ((int)frames.size() >= MAX_FRAMES) break;
    }
}

/* ------------------------ Normalization helper ------------------------ */
void normalize_feature_vectors(vector< vector<double> > &all_feats, double *mean_out, double *std_out) {
    // all_feats: list of vectors length=dim (2)
    int dim = 2;
    for (int d=0; d<dim; d++) mean_out[d] = std_out[d] = 0.0;
    long count = 0;
    for (size_t i=0;i<all_feats.size();i++) {
        for (int d=0; d<dim; d++) mean_out[d] += all_feats[i][d];
        count++;
    }
    if (count == 0) { for (int d=0;d<dim;d++) { mean_out[d]=0; std_out[d]=1; } return; }
    for (int d=0; d<dim; d++) mean_out[d] /= (double)count;
    for (size_t i=0;i<all_feats.size();i++) {
        for (int d=0; d<dim; d++) {
            double diff = all_feats[i][d] - mean_out[d];
            std_out[d] += diff * diff;
        }
    }
    for (int d=0; d<dim; d++) {
        std_out[d] = sqrt( std_out[d] / (double)count );
        if (std_out[d] <= 1e-6) std_out[d] = 1.0;
    }
    // apply normalization inplace
    for (size_t i=0;i<all_feats.size();i++) {
        for (int d=0; d<dim; d++) all_feats[i][d] = (all_feats[i][d] - mean_out[d]) / std_out[d];
    }
}

/* ------------------------ LBG (k-means) vector quantizer ------------------------ */
double euclid_dist2(const double *a, const double *b, int dim) {
    double s=0;
    for (int i=0;i<dim;i++){ double v=a[i]-b[i]; s+=v*v; }
    return s;
}

void lbg_kmeans(const vector< vector<double> > &data, int codebook_size, vector< vector<double> > &centroids) {
    // data: N x dim (dim=2). centroids: codebook_size x dim
    int dim = 2;
    centroids.clear();
    if (data.empty()) return;
    // initialize: pick codebook_size random samples (or evenly spaced)
    int N = (int)data.size();
    for (int k=0;k<codebook_size;k++) {
        int idx = (k * N) / codebook_size;
        vector<double> c(dim);
        for (int d=0; d<dim; d++) c[d] = data[idx][d];
        centroids.push_back(c);
    }
    // k-means iterations
    int maxiter = KMEANS_ITERS;
    vector<int> assign(N, 0);
    for (int it=0; it<maxiter; it++) {
        bool changed = false;
        // assign
        for (int i=0;i<N;i++){
            double best = 1e300; int bestk = 0;
            for (int k=0;k<codebook_size;k++){
                double sum = 0;
                for (int d=0;d<dim;d++){
                    double diff = data[i][d] - centroids[k][d];
                    sum += diff*diff;
                }
                if (sum < best) { best = sum; bestk = k; }
            }
            if (assign[i] != bestk) { assign[i] = bestk; changed = true; }
        }
        // recompute centroids
        vector< vector<double> > sum(codebook_size, vector<double>(dim, 0.0));
        vector<int> cnt(codebook_size, 0);
        for (int i=0;i<N;i++){
            int k = assign[i];
            cnt[k]++;
            for (int d=0;d<dim;d++) sum[k][d] += data[i][d];
        }
        for (int k=0;k<codebook_size;k++){
            if (cnt[k] > 0) {
                for (int d=0;d<dim;d++) centroids[k][d] = sum[k][d] / cnt[k];
            } else {
                // reinitialize empty centroid from a random data point
                int idx = rand() % N;
                for (int d=0;d<dim;d++) centroids[k][d] = data[idx][d];
            }
        }
        if (!changed) break;
    }
}

/* ------------------------ Quantize vector frame -> symbol ------------------------ */
int quantize_vector_to_symbol(const vector<double> &vec, const vector< vector<double> > &centroids) {
    int bestk = 0;
    double best = 1e300;
    int dim = (int)vec.size();
    for (size_t k=0;k<centroids.size();k++){
        double s=0;
        for (int d=0; d<dim; d++){
            double diff = vec[d] - centroids[k][d];
            s += diff*diff;
        }
        if (s < best) { best = s; bestk = (int)k; }
    }
    return bestk;
}

/* ------------------------ HMM code (forward + Baum-Welch) using static arrays ------------------------ */
static double alpha[MAX_SEQ_LEN][STATES];
static double beta_mat[MAX_SEQ_LEN][STATES];
static double gamma_mat[MAX_SEQ_LEN][STATES];
static double xi_mat[MAX_SEQ_LEN][STATES][STATES];
static double scale_arr[MAX_SEQ_LEN];

double forward_scaled_loglik(const HMM &hmm, const int *O, int T) {
    if (T <= 0) return -1e300;
    int N = hmm.N;
    double c0 = 0;
    for (int i=0;i<N;i++) { alpha[0][i] = hmm.pi[i] * hmm.B[i][ O[0] ]; c0 += alpha[0][i]; }
    if (c0 <= 0) c0 = 1e-300;
    scale_arr[0] = c0;
    for (int i=0;i<N;i++) alpha[0][i] /= c0;
    for (int t=1;t<T;t++){
        double ct = 0;
        int ot = O[t];
        for (int j=0;j<N;j++){
            double s = 0;
            for (int i=0;i<N;i++) s += alpha[t-1][i] * hmm.A[i][j];
            alpha[t][j] = s * hmm.B[j][ot];
            ct += alpha[t][j];
        }
        if (ct <= 0) ct = 1e-300;
        scale_arr[t] = ct;
        for (int j=0;j<N;j++) alpha[t][j] /= ct;
    }
    double loglik = 0;
    for (int t=0;t<T;t++) loglik += log(scale_arr[t]);
    return -loglik;
}

void baum_welch_train(HMM &hmm, int **seqs, int *seq_lens, int S) {
    int N = hmm.N;
    int M = hmm.M;
    if (S <= 0) return;
    for (int iter=0; iter<EM_ITERS; iter++) {
        double A_num[STATES][STATES]; for (int i=0;i<N;i++) for (int j=0;j<N;j++) A_num[i][j]=0;
        double A_den[STATES]; for (int i=0;i<N;i++) A_den[i]=0;
        double B_num[STATES][SYMBOLS]; for (int i=0;i<N;i++) for (int k=0;k<M;k++) B_num[i][k]=0;
        double B_den[STATES]; for (int i=0;i<N;i++) B_den[i]=0;
        double pi_acc[STATES]; for (int i=0;i<N;i++) pi_acc[i]=0;

        for (int s=0; s<S; s++) {
            int *O = seqs[s];
            int T = seq_lens[s];
            if (T <= 0) continue;
            // forward
            double c0=0;
            for (int i=0;i<N;i++) { alpha[0][i] = hmm.pi[i] * hmm.B[i][ O[0] ]; c0+=alpha[0][i]; }
            if (c0 <= 0) c0=1e-300;
            scale_arr[0]=c0;
            for (int i=0;i<N;i++) alpha[0][i] /= c0;
            for (int t=1;t<T;t++){
                double ct=0; int ot = O[t];
                for (int j=0;j<N;j++){
                    double ssum=0;
                    for (int i=0;i<N;i++) ssum += alpha[t-1][i]*hmm.A[i][j];
                    alpha[t][j] = ssum * hmm.B[j][ot];
                    ct += alpha[t][j];
                }
                if (ct <= 0) ct=1e-300;
                scale_arr[t]=ct;
                for (int j=0;j<N;j++) alpha[t][j] /= ct;
            }
            // backward
            for (int i=0;i<N;i++) beta_mat[T-1][i] = 1.0 / scale_arr[T-1];
            for (int t=T-2;t>=0;t--){
                int ot1 = O[t+1];
                for (int i=0;i<N;i++){
                    double ssum=0;
                    for (int j=0;j<N;j++) ssum += hmm.A[i][j] * hmm.B[j][ot1] * beta_mat[t+1][j];
                    beta_mat[t][i] = ssum / scale_arr[t];
                }
            }
            // gamma & xi
            for (int t=0;t<T-1;t++){
                double denom=0; int ot1=O[t+1];
                for (int i=0;i<N;i++) for (int j=0;j<N;j++) denom += alpha[t][i] * hmm.A[i][j] * hmm.B[j][ot1] * beta_mat[t+1][j];
                if (denom <= 0) denom = 1e-300;
                for (int i=0;i<N;i++){
                    gamma_mat[t][i] = 0.0;
                    for (int j=0;j<N;j++){
                        double val = (alpha[t][i] * hmm.A[i][j] * hmm.B[j][ot1] * beta_mat[t+1][j]) / denom;
                        xi_mat[t][i][j] = val;
                        gamma_mat[t][i] += val;
                        A_num[i][j] += val;
                        A_den[i] += val;
                    }
                }
            }
            double denomLast = 0;
            for (int i=0;i<N;i++) denomLast += alpha[T-1][i];
            if (denomLast <= 0) denomLast = 1e-300;
            for (int i=0;i<N;i++) gamma_mat[T-1][i] = alpha[T-1][i] / denomLast;
            for (int i=0;i<N;i++) pi_acc[i] += gamma_mat[0][i];
            for (int t=0;t<T;t++){
                int ot = O[t];
                for (int i=0;i<N;i++) {
                    B_num[i][ot] += gamma_mat[t][i];
                    B_den[i] += gamma_mat[t][i];
                }
            }
        } // sequences
        // re-estimate
        for (int i=0;i<N;i++) {
            hmm.pi[i] = pi_acc[i] / (double)S;
            if (hmm.pi[i] < 1e-12) hmm.pi[i] = 1e-12;
        }
        normalize_arr(hmm.pi, N);
        for (int i=0;i<N;i++){
            double denom = 0;
            for (int j=0;j<N;j++) denom += A_num[i][j];
            if (denom <= 0) denom = 1e-300;
            for (int j=0;j<N;j++){
                hmm.A[i][j] = A_num[i][j] / denom;
                if (hmm.A[i][j] < 1e-12) hmm.A[i][j] = 1e-12;
            }
            normalize_arr(hmm.A[i], N);
        }
        for (int i=0;i<N;i++){
            double denom = B_den[i];
            if (denom <= 0) denom = 1e-300;
            for (int k=0;k<M;k++){
                hmm.B[i][k] = B_num[i][k] / denom;
                if (hmm.B[i][k] < 1e-12) hmm.B[i][k] = 1e-12;
            }
            normalize_arr(hmm.B[i], M);
        }
    } // iter
}

/* ------------------------ Save model as matrices (pi, A, B) ------------------------ */
bool ensure_dir_exists(const string &dir) {
#ifdef _WIN32
    _mkdir(dir.c_str());
    return true;
#else
    mkdir(dir.c_str(), 0755);
    return true;
#endif
}

bool save_model_text(const HMM &hmm, const string &path) {
    FILE *f = fopen(path.c_str(), "wt");
    if (!f) return false;
    fprintf(f, "%d %d\n", hmm.N, hmm.M);
    // pi
    for (int i=0;i<hmm.N;i++) {
        if (i) fprintf(f, " ");
        fprintf(f, "%.12g", hmm.pi[i]);
    }
    fprintf(f, "\n");
    // A
    for (int i=0;i<hmm.N;i++) {
        for (int j=0;j<hmm.N;j++) {
            if (j) fprintf(f, " ");
            fprintf(f, "%.12g", hmm.A[i][j]);
        }
        fprintf(f, "\n");
    }
    // B
    for (int i=0;i<hmm.N;i++) {
        for (int k=0;k<hmm.M;k++) {
            if (k) fprintf(f, " ");
            fprintf(f, "%.12g", hmm.B[i][k]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return true;
}

/* ------------------------ Save codebook ------------------------ */
bool save_codebook(const vector< vector<double> > &centroids, const string &path) {
    FILE *f = fopen(path.c_str(), "wt");
    if (!f) return false;
    int K = (int)centroids.size();
    int dim = (K>0) ? (int)centroids[0].size() : 0;
    fprintf(f, "%d %d\n", K, dim);
    for (int k=0;k<K;k++){
        for (int d=0; d<dim; d++) {
            if (d) fprintf(f, " ");
            fprintf(f, "%.12g", centroids[k][d]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return true;
}

/* ------------------------ Main single-run pipeline ------------------------ */
int main(int argc, char **argv) {
    if (argc >= 2) {
        string arg = argv[1];
        if (arg.size() > 0) {
            char last = arg[arg.size()-1];
            if (last != '\\' && last != '/') arg += "\\";
        }
        DATA_FOLDER = arg;
    }
    if (MODELS_DIR.size() == 0) {
        MODELS_DIR = DATA_FOLDER;
        // append models dir
#ifdef _WIN32
        MODELS_DIR += "models\\";
#else
        MODELS_DIR += "models/";
#endif
    }
    cout << "DATA_FOLDER = " << DATA_FOLDER << "\n";
    cout << "MODELS_DIR = " << MODELS_DIR << "\n";
    srand(RANDOM_SEED);

    // 1) Load training files (1..30), extract frame features (STE,ZCR) and accumulate for codebook
    vector< vector<double> > all_train_frames; // flattened list of all frames for kmeans
    vector< vector< vector<double> > > train_frames_by_digit(10); // per digit list of frame vectors

   // allocate samples on heap instead of stack to avoid stack overflow
double *samples = (double*) malloc(sizeof(double) * MAX_SAMPLES);
if (!samples) {
    std::cerr << "ERROR: failed to allocate samples buffer of size " << MAX_SAMPLES << "\n";
    return 1;
}
int nsamp = 0;

    for (int d=0; d<10; d++) {
        for (int occ=1; occ<=30; occ++) {
            char fname[1024];
            sprintf(fname, "%s254101058_E_%d_%d.txt", DATA_FOLDER.c_str(), d, occ);
            int n = load_samples_txt(fname, samples, nsamp);
            if (n <= 0) {
                cout << "Warning: missing/empty train file: " << fname << "\n";
                continue;
            }
            vector< vector<double> > frames;
            compute_frames_ste_zcr(samples, n, frames);
            // add frames to both global list and per-digit
            for (size_t i=0;i<frames.size();i++) {
                all_train_frames.push_back(frames[i]);
                train_frames_by_digit[d].push_back(frames[i]);
            }
        }
        cout << "Digit " << d << " training frames = " << train_frames_by_digit[d].size() << "\n";
    }

    if (all_train_frames.empty()) { cout << "No training frames found. Exiting.\n"; return 2; }

    // 2) Normalize feature vectors (global mean/std) - compute mean/std from all_train_frames
    double meanv[2], stdv[2];
    normalize_feature_vectors(all_train_frames, meanv, stdv);
    // apply same normalization to train_frames_by_digit
    for (int d=0; d<10; d++) {
        for (size_t i=0;i<train_frames_by_digit[d].size();i++)
            for (int k=0;k<2;k++)
                train_frames_by_digit[d][i][k] = (train_frames_by_digit[d][i][k] - meanv[k]) / stdv[k];
    }

    // 3) Build codebook via LBG/k-means on all frames (normalized)
    // build a flattened normalized vector list from all_train_frames now normalized by earlier function
    // We already modified all_train_frames inside normalize_feature_vectors.
    vector< vector<double> > centroids;
    lbg_kmeans(all_train_frames, SYMBOLS, centroids);
    if (centroids.size() != SYMBOLS) cout << "Warning: codebook size != SYMBOLS\n";

    // save codebook
    ensure_dir_exists(MODELS_DIR);
    char codepath[1024];
    sprintf(codepath, "%scodebook.txt", MODELS_DIR.c_str());
    if (save_codebook(centroids, codepath)) cout << "Saved codebook: " << codepath << "\n";
    else cout << "Failed to save codebook\n";

    // 4) Convert each training utterance into discrete symbol sequences (per-digit arrays)
    int *seq_ptrs[10][40];
    int seq_lens[10][40];
    int seq_counts[10];
    for (int d=0; d<10; d++) seq_counts[d]=0;
    for (int d=0; d<10; d++) {
        // we need to re-load each file frames, normalize by mean/std and quantize
        for (int occ=1; occ<=30; occ++) {
            char fname[1024];
            sprintf(fname, "%s254101058_E_%d_%d.txt", DATA_FOLDER.c_str(), d, occ);
            int n = load_samples_txt(fname, samples, nsamp);
            if (n <= 0) continue;
            vector< vector<double> > frames;
            compute_frames_ste_zcr(samples, n, frames);
            if (frames.empty()) continue;
            int L = (int)frames.size();
            int *obs = (int*) malloc(sizeof(int) * L);
            for (int i=0;i<L;i++) {
                // normalize by mean/std
                vector<double> v = frames[i];
                for (int k=0;k<2;k++) v[k] = (v[k] - meanv[k]) / stdv[k];
                obs[i] = quantize_vector_to_symbol(v, centroids);
            }
            seq_ptrs[d][ seq_counts[d] ] = obs;
            seq_lens[d][ seq_counts[d] ] = L;
            seq_counts[d]++;
        }
        cout << "Digit " << d << " training sequences (frames->symbols) = " << seq_counts[d] << "\n";
    }

    // 5) Train HMM per digit, save models (pi, A, B)
    HMM models[10];
    for (int d=0; d<10; d++) {
        models[d].N = STATES; models[d].M = SYMBOLS;
        init_hmm_left_right(models[d]);
        int S = seq_counts[d];
        if (S <= 0) {
            cout << "Warning: no training sequences for digit " << d << "\n";
            continue;
        }
        int **seqs = (int**) malloc(sizeof(int*) * S);
        int *lens = (int*) malloc(sizeof(int) * S);
        for (int i=0;i<S;i++) { seqs[i] = seq_ptrs[d][i]; lens[i] = seq_lens[d][i]; }
        cout << "Training HMM for digit " << d << " with " << S << " sequences ...\n";
        baum_welch_train(models[d], seqs, lens, S);
        // save model to models/digit_d.txt
        char mpath[1024];
#ifdef _WIN32
        sprintf(mpath, "%sdigit_%d.txt", MODELS_DIR.c_str(), d);
#else
        sprintf(mpath, "%sdigit_%d.txt", MODELS_DIR.c_str(), d);
#endif
        if (save_model_text(models[d], mpath)) cout << "Saved model: " << mpath << "\n";
        else cout << "Failed to save model: " << mpath << "\n";
        free(seqs); free(lens);
    }

    // 6) Load test files (31..40), extract frames, normalize, quantize, run forward against all models
    cout << "\n=== TEST RESULTS (one line per test file) ===\n";
    int total = 0, correct = 0;
    for (int d=0; d<10; d++) {
        for (int occ=31; occ<=40; occ++) {
            char fname[1024];
            sprintf(fname, "%s254101058_E_%d_%d.txt", DATA_FOLDER.c_str(), d, occ);
            int n = load_samples_txt(fname, samples, nsamp);
            if (n <= 0) { /* missing test file */ continue; }
            vector< vector<double> > frames;
            compute_frames_ste_zcr(samples, n, frames);
            if (frames.empty()) { continue; }
            int T = (int)frames.size();
            int *obs = (int*) malloc(sizeof(int) * T);
            for (int i=0;i<T;i++) {
                vector<double> v = frames[i];
                for (int k=0;k<2;k++) v[k] = (v[k] - meanv[k]) / stdv[k];
                obs[i] = quantize_vector_to_symbol(v, centroids);
            }
            // test
            double bestScore = -1e300;
            int bestLabel = -1;
            for (int m=0; m<10; m++) {
                double neglog = forward_scaled_loglik(models[m], obs, T); // negative log-likelihood
                double lik = -neglog;
                if (lik > bestScore) { bestScore = lik; bestLabel = m; }
            }
            // print one-line result
            printf("[TEST] %s  -> Predicted=%d  Actual=%d  Score=%.6g  Frames=%d\n", fname, bestLabel, d, bestScore, T);
            total++;
            if (bestLabel == d) correct++;
            free(obs);
        }
    }

    double accuracy = (total>0) ? (100.0 * correct / (double)total) : 0.0;
    printf("\nFINAL ACCURACY = %.2f %%  (%d / %d)\n", accuracy, correct, total);

    // keep console open
    cout << "\nPress ENTER to exit...";
    cin.ignore();
    cin.get();
	// free heap memory
free(samples);

    return 0;
}
