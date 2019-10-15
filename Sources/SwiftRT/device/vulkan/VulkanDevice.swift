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
import CVulkan

public class VulkanDevice : LocalComputeDevice {
    //--------------------------------------------------------------------------
    // properties
    public private(set) var trackingId = 0
    public private (set) weak var service: ComputeService!
    public var deviceArrayReplicaKey = Platform.nextUniqueDeviceId
    public let id: Int
    public var logInfo: LogInfo
    public let name: String
    private let streamId = AtomicCounter(value: -1)
    public var timeout: TimeInterval?
    public let memoryAddressing: MemoryAddressing
    public var deviceErrorHandler: DeviceErrorHandler?
    public var limits: DeviceLimits
    public var _lastError: Error? = nil
    public var _errorMutex: Mutex = Mutex()

    // TODO this should be currently available and not physicalMemory
    public lazy var availableMemory: UInt64 = {
        return ProcessInfo.processInfo.physicalMemory
    }()

    //--------------------------------------------------------------------------
    // initializers
    public init(service: VulkanService,
                physicalDevice: VkPhysicalDevice,
                deviceId: Int,
                logInfo: LogInfo,
                timeout: TimeInterval?) throws
    {
        self.logInfo = logInfo
        self.id = deviceId
        self.service = service
        self.timeout = timeout
        self.memoryAddressing = .discreet // TODO: query this
        
        // query limits
        var deviceProperties = VkPhysicalDeviceProperties()
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties)
        self.limits = DeviceLimits(limits: deviceProperties.limits)
        
        // get the device name
        let deviceNamePointer =
            withUnsafeBytes(of: &deviceProperties.deviceName) {
            return $0.baseAddress!.assumingMemoryBound(to: CChar.self)
        }
        self.name = String(cString: deviceNamePointer)
        
        // devices are statically held by the Platform.service
        trackingId = ObjectTracker.global
                .register(self, namePath: logNamePath, isStatic: true)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }

    //--------------------------------------------------------------------------
    // createArray
    public func createArray(count: Int) throws -> DeviceArray {
//        return try CudaDeviceArray(device: self, count: count)
        fatalError()
    }

    //--------------------------------------------------------------------------
    // createReferenceArray
    /// creates a read only device array from a uma buffer
    public func createReferenceArray(buffer: UnsafeRawBufferPointer)
        -> DeviceArray
    {
        //        return CudaDeviceArray(device: self, buffer: buffer)
        fatalError()
    }

    //--------------------------------------------------------------------------
    // createMutableReferenceArray
    /// creates a read write device array from a uma buffer
    public func createMutableReferenceArray(
            buffer: UnsafeMutableRawBufferPointer) -> DeviceArray
    {
//        return CudaDeviceArray(device: self, buffer: buffer)
        fatalError()
    }

    //--------------------------------------------------------------------------
    // createStream
    public func createQueue(name: String, isStatic: Bool) throws -> DeviceQueue
    {
        fatalError()
    }
}

//==============================================================================
public extension DeviceLimits {
    init(limits: VkPhysicalDeviceLimits) {
        maxComputeSharedMemorySize = Int(limits.maxComputeSharedMemorySize)
        maxComputeWorkGroupCount =
            (Int(limits.maxComputeWorkGroupCount.0),
             Int(limits.maxComputeWorkGroupCount.1),
             Int(limits.maxComputeWorkGroupCount.2))
        maxComputeWorkGroupInvocations =
            Int(limits.maxComputeWorkGroupInvocations)
        maxComputeWorkGroupSize =
            (Int(limits.maxComputeWorkGroupSize.0),
             Int(limits.maxComputeWorkGroupSize.1),
             Int(limits.maxComputeWorkGroupSize.2))
        maxMemoryAllocationCount = Int(limits.maxMemoryAllocationCount)
    }
}
