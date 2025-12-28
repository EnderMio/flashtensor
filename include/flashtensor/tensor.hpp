#include <cstddef>
#include <cstdint>
#include <numeric>
#include <sys/types.h>
#include <utility>
#include <vector>
#include <stdexcept>
#include "storage.hpp"

using DimVector = std::vector<int64_t>;

template<typename T>
class Tensor {
public:
    Tensor(DimVector shape, DeviceType device = DeviceType::cpu)
        : shape_{std::move(shape)}, 
          strides_{shape_.size()}, 
          storage_{compute_size(shape_), device}, 
          is_contiguous_{true}
    { compute_strides(); }
    Tensor(DimVector shape, DimVector strides, Storage<T> storage, int64_t offset)
        : shape_{std::move(shape)}, 
          strides_{std::move(strides)}, 
          storage_{std::move(storage)},
          offset_{offset}, 
          is_contiguous_{check_contiguous()}
    { }
    Tensor(const Tensor<T>& other) 
        : shape_{other.shape_},
          strides_{other.strides_},
          storage_{other.storage_},
          offset_{other.offset_},
          is_contiguous_{other.is_contiguous_}
    { }
    Tensor(Tensor<T>&& other) noexcept
        : shape_{std::move(other.shape_)}, 
          strides_{std::move(other.strides_)}, 
          storage_{std::move(other.storage_)},
          offset_{other.offset_}, 
          is_contiguous_{other.is_contiguous_}
    { }
    Tensor<T>& operator=(const Tensor& other){
        if(this != &other) {
            shape_ = other.shape_;
            strides_ = other.strides_;
            storage_ = other.storage_;
            offset_ = other.offset_;
            is_contiguous_ = other.is_contiguous_;
        }
        return *this;
    }
    Tensor<T>& operator=(Tensor&& other){
        if(this != &other) {
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            storage_ = std::move(other.storage_);
            offset_ = other.offset_;
            is_contiguous_ = other.is_contiguous_;
        }
        return *this;
    }
    Tensor clone() const;
    template<typename... Args>
    decltype(auto) operator()(Args... indices) {
        if (sizeof...(Args) != shape_.size()) {
            throw std::runtime_error("Dimension mismatch");
        }
        return storage_.data()[offset_ + get_offset(std::make_index_sequence<sizeof...(Args)>{}, indices...)];
    }
    template<typename... Args>
    decltype(auto) operator()(Args... indices) const {
        if (sizeof...(Args) != shape_.size()) {
            throw std::runtime_error("Dimension mismatch");
        }
        return storage_.data()[offset_ + get_offset(std::make_index_sequence<sizeof...(Args)>{}, indices...)];
    }

    auto is_contiguous() { return is_contiguous_; }

private:
    DimVector shape_;
    DimVector strides_;
    Storage<T> storage_;
    int64_t offset_ = 0;
    bool is_contiguous_;

    static auto compute_size(const DimVector& shape_) {
        return std::accumulate(shape_.begin(), shape_.end(), int64_t{1}, std::multiplies<int64_t>());
    }

    void compute_strides() noexcept;

    template<size_t... Is, typename... Args>
    int64_t get_offset(std::index_sequence<Is...>, Args... indices) const {
        return ( int64_t{0} + ... + (static_cast<int64_t>(indices) * strides_[Is]) );
    }

    bool check_contiguous() {
        int64_t z = 1;
        for (int64_t i = shape_.size() - 1; i >= 0; --i) {
            if (shape_[i] != 1) {
                if (strides_[i] != z) {
                    return false;
                }
                z *= shape_[i];
            }
        }
        return true;
    }
};

template<typename T>
void Tensor<T>::compute_strides() noexcept {
    std::exclusive_scan(shape_.rbegin(), shape_.rend(), strides_.rbegin(), 1, std::multiplies<int64_t>());
}