#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/penlu_layer.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void PENLUForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const int div_factor, const Dtype* alpha, const Dtype* beta , const Dtype* eta) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
 //   out[index] = in[index] : ( ( pow( (std::max(in[index], Dtype(0)),eta[c] ) ) * ( in[index] > 0 ) + ( ( exp( beta[c]*std::min(in[index], Dtype(0)) ) - 1 ) * alpha[c] ) * (in[index] <= 0) );
    if (in[index] > 0){
        out[index] = (pow(in[index],eta[c]));
        }
    else{
        out[index] = (( exp(beta[c]*in[index] ) - 1 ) * alpha[c]);
        }
  }
}

// CUDA kernel for bottom backward
// template <typename Dtype>
// __global__ void M2PELUBackward(const int n, const int channels, const int dim,
//     const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
//     const int div_factor,
//     const Dtype* top_data, const Dtype* alpha, const Dtype* beta, const Dtype* gamma) {
//   CUDA_KERNEL_LOOP(index, n) {
//     int c = (index / dim) % channels / div_factor;
//     out_diff[index] = in_diff[index] * ((in_data[index] > 0)
//         + (in_data[index] <= 0) * beta[c] * ( top_data[index] + alpha[c]*exp(gamma[c]) ) );
//   }
// }

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void PENLUParamBackward(const int n, 
    const int rows, const int rowPitch, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff_alpha, Dtype* out_diff_beta, Dtype* out_diff_eta,
    const int channels, const int dim, const int div_factor, const Dtype* top_data, Dtype* out_diff,
    const Dtype* alpha, const Dtype* beta, const Dtype* eta ) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff_alpha[index] = in_diff[index] * ( exp(beta[c]*in_data[index] ) - 1 ) * (in_data[index] <= 0);
    out_diff_beta[index]  = in_diff[index] * ( in_data[index] * (top_data[index] + alpha[c]  ) ) * (in_data[index] <= 0);
    out_diff_eta[index]  = in_diff[index] * ( top_data[index]*log(in_data[index]) ) * (in_data[index] > 0);
    out_diff[index] = in_diff[index] * ( (eta[c]*(top_data[index]/in_data[index])) * (in_data[index] > 0) + (in_data[index] <= 0) * beta[c] * ( top_data[index] + alpha[c] ) );
    for ( int k = 1; k < rows; k++ ) {
    	int tmp_index = index + k*rowPitch;
      out_diff_alpha[index] += in_diff[tmp_index] * (  exp(beta[c] * in_data[tmp_index] ) - 1  )* (in_data[tmp_index] <= 0);
      out_diff_beta[index]  += in_diff[tmp_index] * (  in_data[tmp_index] * (top_data[tmp_index] + alpha[c] ) ) * (in_data[tmp_index] <= 0);
      out_diff_eta[index]  += in_diff[tmp_index] * (  top_data[tmp_index] * log(in_data[tmp_index]) * (in_data[tmp_index] > 0) );
      out_diff[tmp_index] = in_diff[tmp_index] * ( (eta[c]*top_data[tmp_index]/in_data[tmp_index]) * (in_data[tmp_index] > 0) + (in_data[tmp_index] <= 0) * beta[c] * ( top_data[tmp_index] + alpha[c] ) );
    }
  }
}

template <typename Dtype>
void PENLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const int div_factor = channel_shared_ ? channels : 1;

  const Dtype* alpha = this->blobs_[0]->gpu_data();
  const Dtype* beta  = this->blobs_[1]->gpu_data();
  const Dtype* eta  = this->blobs_[2]->gpu_data();

  // For in-place computation
  if (top[0] == bottom[0]) {
    // exit(0);
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  PENLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, div_factor, alpha, beta, eta);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void PENLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const int div_factor = channel_shared_ ? channels : 1;

  const Dtype* alpha = this->blobs_[0]->gpu_data();
  const Dtype* beta  = this->blobs_[1]->gpu_data();
  const Dtype* eta  = this->blobs_[2]->gpu_data();

  // For in-place computation
  if (top[0] == bottom[0]) {
    // exit(0);
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to param alpha
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
//  if (this->param_propagate_down_[0]) {
  if (1) {
    Dtype* alpha_diff = this->blobs_[0]->mutable_gpu_diff();
    Dtype* beta_diff  = this->blobs_[1]->mutable_gpu_diff();
    Dtype* eta_diff  = this->blobs_[2]->mutable_gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    int cdim = channels * dim;

    // compute element-wise diff
    // NOLINT_NEXT_LINE(whitespace/operators)
    PENLUParamBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
      CAFFE_CUDA_NUM_THREADS>>>(
      cdim, bottom[0]->num(), top[0]->offset(1), top_diff ,
      bottom_data ,
      backward_buff_alpha.mutable_gpu_diff(), backward_buff_beta.mutable_gpu_diff(), backward_buff_eta.mutable_gpu_diff(), 
      channels, dim, div_factor, top_data, bottom_diff, alpha, beta, eta);
    CUDA_POST_KERNEL_CHECK;
    if (channel_shared_) {
      Dtype dsum_alpha;
      Dtype dsum_beta;
      Dtype dsum_eta;
      caffe_gpu_dot<Dtype>(channels * dim, backward_buff_alpha.gpu_diff(),
       multiplier_.gpu_data(), &dsum_alpha);
      caffe_gpu_dot<Dtype>(channels * dim, backward_buff_beta.gpu_diff(),
       multiplier_.gpu_data(), &dsum_beta);
      caffe_gpu_dot<Dtype>(channels * dim, backward_buff_eta.gpu_diff(),
       multiplier_.gpu_data(), &dsum_eta);
      caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum_alpha), alpha_diff);
      caffe_gpu_add_scalar(this->blobs_[1]->count(), Dtype(dsum_beta),  beta_diff );
      caffe_gpu_add_scalar(this->blobs_[2]->count(), Dtype(dsum_eta),  eta_diff );
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_buff_alpha.gpu_diff(), multiplier_.gpu_data(), 1.,
        alpha_diff);
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_buff_beta.gpu_diff(), multiplier_.gpu_data(), 1.,
        beta_diff);
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_buff_eta.gpu_diff(), multiplier_.gpu_data(), 1.,
        eta_diff);
    }
  }

  // Propagate to bottom
//   if (propagate_down[0]) {
//     Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
//     // const Dtype* slope_data = this->blobs_[0]->gpu_data();
//     // int div_factor = channel_shared_ ? channels : 1;
//     // NOLINT_NEXT_LINE(whitespace/operators)
//     M2PELUBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
//         CAFFE_CUDA_NUM_THREADS>>>(
//         count, channels, dim, top_diff, bottom_data, bottom_diff, div_factor,
//         top_data, alpha, beta, gamma);
//     CUDA_POST_KERNEL_CHECK;
//   }
}


INSTANTIATE_LAYER_GPU_FUNCS(PENLULayer);


}  // namespace caffe
