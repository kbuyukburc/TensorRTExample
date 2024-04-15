#include "gelu.h"
#include <math.h>

namespace Tn {
    template<typename T>
    void write(char*& buffer, const T& val) {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T>
    void read(const char*& buffer, T& val) {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}  

namespace nvinfer1
{
    gelu::gelu()
    {
    }

    gelu::~gelu()
    {
    }
    // create the plugin at runtime from a byte stream
    gelu::gelu(const void* data, size_t length)
    {
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        Tn::read(d, mInputSize);
        assert(d == a + length);
    }

    void gelu::serialize(void* buffer) const noexcept
    {
        char* d = static_cast<char*>(buffer), *a = d;
        Tn::write(d, mInputSize);
        assert(d == a + getSerializationSize());
    }

    size_t gelu::getSerializationSize() const noexcept
    {
        return sizeof(mInputSize);
    }

    int gelu::initialize() noexcept
    {
        return 0;
    }

    Dims gelu::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)  noexcept
    {
        assert(nbInputDims == 1);
        Dims outputDims;
        outputDims.nbDims = inputs[0].nbDims;
        for (int i = 0; i < inputs[0].nbDims; i++) {
            outputDims.d[i] = inputs[0].d[i];
        }
        return outputDims;
    }

    // Set plugin namespace
    void gelu::setPluginNamespace(const char* pluginNamespace) noexcept
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* gelu::getPluginNamespace() const noexcept
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType gelu::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const  noexcept
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool gelu::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool gelu::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
    {
        return false;
    }

    void gelu::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) noexcept
    {

        mInputSize = 1;
        for (int i = 0; i < in[0].dims.nbDims; i++) {
            mInputSize *= in[0].dims.d[i];
        }
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void gelu::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
    {
    }

    // Detach the plugin object from its execution context.
    void gelu::detachFromContext() noexcept {}

    const char* gelu::getPluginType() const noexcept
    {
        return "geluLayer_TRT";
    }

    const char* gelu::getPluginVersion() const noexcept
    {
        return "1";
    }

    void gelu::destroy() noexcept
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* gelu::clone() const noexcept
    {
        gelu *p = new gelu();
        p->setPluginNamespace(mPluginNamespace);
        p->setInputSize(mInputSize);
        return p;
    }

    __global__ void geluKer(const float *in, float *out, int size) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= size)
            return;
        // x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        out[idx] = in[idx] * 0.5 *(1.0 + erf(in[idx]/1.4142135381698608));
        // out[idx] = 77;
    }
    void gelu::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {

        int numElem = batchSize * mInputSize;
        geluKer<<<(numElem + mThreadCount - 1) / mThreadCount, mThreadCount>>>
            (inputs[0], output, numElem);
    }

    int gelu::enqueue(int batchSize, void const * const * inputs, void* const * outputs, void* workspace, cudaStream_t stream) noexcept
    {
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection geluCreator::mFC{};
    std::vector<PluginField> geluCreator::mPluginAttributes;

    geluCreator::geluCreator()
    {
        mPluginAttributes.clear();
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* geluCreator::getPluginName() const noexcept
    {
            return "geluLayer_TRT";
    }

    const char* geluCreator::getPluginVersion() const noexcept
    {
            return "1";
    }

    const PluginFieldCollection* geluCreator::getFieldNames() noexcept
    {
            return &mFC;
    }

    IPluginV2IOExt* geluCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
    {
        gelu* obj = new gelu();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* geluCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
    {
        // This object will be deleted when the network is destroyed, which will
        gelu* obj = new gelu(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }


}

