//
//  MetalRNNSequenceGRU.mm
//  MNN
//
//  Created for VOICEVOX MNN Metal GPU Support
//  Copyright © 2024
//
//  Note: GRU is a complex sequential operation that doesn't benefit significantly
//  from GPU parallelization. This implementation returns nullptr to delegate
//  execution to the CPU backend, which is actually more efficient for this
//  operation type. The MNN scheduler will automatically use CPU for GRU
//  while still accelerating other operations (MatMul, Conv, etc.) on GPU.
//

#import "backend/metal/MetalBackend.hpp"
#import "core/Macro.h"
#import "MNN_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

// Metal GRU Creator - delegates to CPU for better performance
// GRU operations are inherently sequential and don't parallelize well on GPU
class MetalRNNSequenceGRUCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op,
                                Backend *backend, const std::vector<Tensor *>& outputs) const override {
        // Return nullptr to delegate to CPU backend
        // This is intentional - GRU's sequential nature makes it more efficient on CPU
        // while other operations (MatMul, Conv, etc.) still benefit from Metal GPU
        return nullptr;
    }
};

REGISTER_METAL_OP_CREATOR(MetalRNNSequenceGRUCreator, OpType_RNNSequenceGRU);

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
