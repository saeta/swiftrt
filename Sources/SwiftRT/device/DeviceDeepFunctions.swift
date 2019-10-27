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
import Foundation

//==============================================================================
// ConvolutionProperties
public struct ConvolutionProperties: Codable {
    var activationNan: NanPropagation = .noPropagate
    var activationReluCeiling: Double = 0
    var backwardDataAlgorithm: ConvolutionBwdDataAlgorithm = .fastest
    var backwardDataWorkspaceLimit: Int = 10.MB
    var backwardFilterAlgorithm: ConvolutionBwdFilterAlgorithm = .fastest
    var backwardFilterWorkspaceLimit: Int = 10.MB
    var forwardAlgorithm: ConvolutionFwdAlgorithm = .fastest
    var forwardWorkspaceLimit: Int = 10.MB
    var mode: ConvolutionMode = .crossCorrelation
}

//==============================================================================
// ConvolutionFwdAlgorithm
public enum ConvolutionFwdAlgorithm: Int, Codable, CaseIterable {
    case implicitGEMM
    case implicitPrecompGEMM
    case gemm
    case direct
    case fft
    case fftTiling
    case winograd
    case winogradNonFused
    case deterministic
    case fastest
    case noWorkspace
    case workspaceLimit
}

//==============================================================================
// ConvolutionBwdDataAlgorithm
public enum ConvolutionBwdDataAlgorithm: Int, Codable, CaseIterable {
    case algo0
    case algo1
    case fft
    case fftTiling
    case winograd
    case winogradNonFused
    case deterministic
    case fastest
    case noWorkspace
    case workspaceLimit
}

//==============================================================================
// ConvolutionBwdFilterAlgorithm
public enum ConvolutionBwdFilterAlgorithm: Int, Codable, CaseIterable {
    case algo0
    case algo1
    case algo3
    case fft
    case winograd
    case winogradNonFused
    case numAlgorithms
    case deterministic
    case fastest
    case noWorkspace
    case workspaceLimit
}

//==============================================================================
// ConvolutionMode
public enum ConvolutionMode: Int, Codable, CaseIterable {
    case convolution
    case crossCorrelation
}

//==============================================================================
public enum PoolingMode {
    case averageExcludePadding
    case averageIncludePadding
    case max
    case maxDeterministic
}

//==============================================================================
public enum ActivationMode {
    case sigmoid
    case relu
    case tanh
    case clippedRelu
    case elu
    case identity
}

//==============================================================================
public enum TransposeOp {
    case transpose
    case noTranspose
    case conjugateTranspose
}

//==============================================================================
/// DeviceLimits
/// parameters defining maximum device capabilties
public struct DeviceLimits {
    let maxComputeSharedMemorySize: Int
    let maxComputeWorkGroupCount: (Int, Int, Int)
    let maxComputeWorkGroupInvocations: Int
    let maxComputeWorkGroupSize: (Int, Int, Int)
    let maxMemoryAllocationCount: Int
}

//==============================================================================
public enum SoftmaxAlgorithm {
    case accurate, fast, log
}

//==============================================================================
public enum SoftmaxMode {
    case channel, instance
}

