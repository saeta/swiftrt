//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import CCuda

// *** TODO design questions!
// 1) class or struct, how are things retained, reused, etc?
// 2) should the input be retained to guarentee that init
//    matches the same shape as inferring? Or just assert in inferring?

public final class CudaConvolution<T> where
    T: TensorView, T.Element: AnyFloatingPoint
{
    // properties
    private var zero: T.Element = 0
    private var one: T.Element = 1
    private let poolingDescriptor: PoolingDescriptor
    private let xTensorDescriptor: TensorDescriptor
    private let yTensorDescriptor: TensorDescriptor
    private var xDiff: T!
    private var y: T

    public init(x: T,
                filter: T,
                strides: [Int],
                padding: [Int],
                dilations: [Int],
                properties: ConvolutionProperties)
    {
        fatalError()
    }
}

//==============================================================================
// ConvolutionFwdAlgorithm
extension ConvolutionFwdAlgorithm {
    public var cudnn: cudnnConvolutionFwdAlgo_t {
        get {
            let algs: [ConvolutionFwdAlgorithm: cudnnConvolutionFwdAlgo_t] = [
                .implicitGEMM: CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                .implicitPrecompGEMM: CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                .gemm: CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                .direct: CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
                .fft: CUDNN_CONVOLUTION_FWD_ALGO_FFT,
                .fftTiling: CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
                .winograd: CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
                .winogradNonFused: CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
            ]
            return algs[self]!
        }
    }
}

//==============================================================================
// ConvolutionBwdDataAlgorithm
extension ConvolutionBwdDataAlgorithm {
    public var cudnn: cudnnConvolutionBwdDataAlgo_t {
        get {
            let algs: [ConvolutionBwdDataAlgorithm: cudnnConvolutionBwdDataAlgo_t] = [
            .algo0: CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            .algo1: CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
            .fft: CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
            .fftTiling: CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
            .winograd: CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
            .winogradNonFused: CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
            ]
            return algs[self]!
        }
    }
}

let numConvolutionBwdDataAlgorithms = 6

//==============================================================================
// ConvolutionBwdFilterAlgorithm
extension ConvolutionBwdFilterAlgorithm {
    public var cudnn: cudnnConvolutionBwdFilterAlgo_t {
        get {
            let algs: [ConvolutionBwdFilterAlgorithm: cudnnConvolutionBwdFilterAlgo_t] = [
                .algo0: CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                .algo1: CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                .algo3: CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
                .fft: CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
                .winograd: CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD,
                .winogradNonFused: CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
            ]
            return algs[self]!
        }
    }
}

let numConvolutionBwdFilterAlgorithms = 5

//==============================================================================
// ConvolutionMode
extension ConvolutionMode {
    public var cudnn: cudnnConvolutionMode_t {
        get {
            switch self {
            case .convolution: return CUDNN_CONVOLUTION
            case .crossCorrelation: return CUDNN_CROSS_CORRELATION
            }
        }
    }
}

