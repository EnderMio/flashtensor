#include <memory>
#include <iostream>
#include <stdexcept>
#include <cstddef>

template<typename T>
void cpu_deleter(T* ptr) noexcept {
    delete [] ptr;
}

template<typename T>
auto cpu_allocator(std::size_t size) {
    return std::shared_ptr<T[]>(new T[size], cpu_deleter<T>);
}

template<typename T>
void mock_cuda_deleter(T* ptr) noexcept {
    std::cout << "Cleaning up Mock GPU memory..." << std::endl;
    delete [] ptr;
}

template<typename T>
auto mock_cuda_allocator(std::size_t size) {
    return std::shared_ptr<T[]>(new T[size], mock_cuda_deleter<T>);
}

enum class DeviceType { cpu, cuda };

template<typename T>
class Storage {
public:
    Storage(size_t size, DeviceType device);
    const T* data() const { return data_.get(); }
    T* data() { return data_.get(); }
private:
    DeviceType device_;
    std::size_t size_;
    std::shared_ptr<T[]> data_;
};

template<typename T>
Storage<T>::Storage(size_t size, DeviceType device) : size_{size}, device_{device} {
    if (device_ == DeviceType::cpu) {
        data_ = cpu_allocator<T>(size_);
    } else if (device_ == DeviceType::cuda) {
        data_ = mock_cuda_allocator<T>(size_);
    } else {
        throw std::runtime_error("invalid device type");
    }
};