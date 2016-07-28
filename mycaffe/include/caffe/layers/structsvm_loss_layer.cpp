#ifndef CAFFE_STRUCTSVM_LOSS_LAYER_HPP_
#define CAFFE_STRUCTSVM_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe{

template <typename Dtype>
class StructSVMLossLayer : public LossLayer<Dtype> {
 public:
  explicit StructSVMLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),start(0),end(-1),slabel(-1) {}
  virtual ~StructSVMLossLayer(){
	  r_sum.clear();
	  max_sum.clear();
	  m_begin.clear();
	  m_end.clear();
  }
  void  Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) ;
  virtual inline const char* type() const { return "StructSVMLoss"; }
  //inline const vector<Dtype>& get_max_sum() const { return max_sum; }
  inline const vector<int>&   get_max_begin() const {return m_begin; }
  inline const vector<int>&   get_max_end  () const {return m_end; }
  virtual void Inference (const Dtype* array, const Dtype * label,  int num, int dim, int k);
 protected:
  /// @copydoc StructSVMLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Inference_augmented(const Dtype* array, const Dtype* label, int num, int dim);

  //vector<Dtype> r_sum, max_sum;
  vector<vector<int> > m_begin, m_end;
  //constant margin
  vector<vector<Dtype> > max_sum;
  //vector<vector<std::pair<int,int> > > m_ind;
 private:
  int r_begin;
  int start,end,slabel;
  Blob<Dtype> loss_t;
  vector<Dtype> r_sum;

};

}  // namespace caffe

#endif  // CAFFE_STRUCTSVM_LOSS_LAYER_HPP_