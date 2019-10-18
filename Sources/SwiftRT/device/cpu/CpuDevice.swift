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
import Dispatch

public class CpuDevice: LocalComputeDevice {
    //--------------------------------------------------------------------------
    // properties
    public private(set) var trackingId = 0
    public let deviceArrayReplicaKey = Platform.nextUniqueDeviceId
    public let id: Int
    public var logInfo: LogInfo
    public let name: String
    public weak var service: ComputeService!
    private let queueId = AtomicCounter(value: -1)
    public var timeout: TimeInterval?
    public let memoryAddressing: MemoryAddressing
    public var deviceErrorHandler: DeviceErrorHandler?
    public var limits: DeviceLimits
    public var _lastError: Error? = nil
    public var _errorMutex: Mutex = Mutex()
    public var heaps: [DeviceHeapProperties]
    
    //--------------------------------------------------------------------------
	// initializers
	public init(service: ComputeService,
                deviceId: Int,
                logInfo: LogInfo,
                memoryAddressing: MemoryAddressing,
                timeout: TimeInterval?) {
        self.name = "cpu:\(deviceId)"
		self.logInfo = logInfo
		self.id = deviceId
		self.service = service
        self.timeout = timeout
        self.memoryAddressing = memoryAddressing
        
        // TODO: determine meaningful values, not currently used
        self.limits = DeviceLimits(
            maxComputeSharedMemorySize: 1,
            maxComputeWorkGroupCount: (1, 1, 1),
            maxComputeWorkGroupInvocations: 1,
            maxComputeWorkGroupSize: (1, 1, 1),
            maxMemoryAllocationCount: 1
        )
        
        // TODO:
        self.heaps = []

		// devices are statically held by the Platform.service
        trackingId = ObjectTracker.global
            .register(self, namePath: logNamePath, isStatic: true)
	}
	deinit { ObjectTracker.global.remove(trackingId: trackingId) }

    //--------------------------------------------------------------------------
	// createArray
	//	This creates memory on the device
	public func createArray(count: Int) throws -> DeviceArray {
        return CpuDeviceArray(device: self, count: count)
	}
    
    //--------------------------------------------------------------------------
    // createMutableReferenceArray
    /// creates a device array from a uma buffer.
    public func createMutableReferenceArray(
        buffer: UnsafeMutableRawBufferPointer) -> DeviceArray {
        return CpuDeviceArray(device: self, buffer: buffer)
    }
    
    //--------------------------------------------------------------------------
    // createReferenceArray
    /// creates a device array from a uma buffer.
    public func createReferenceArray(buffer: UnsafeRawBufferPointer)
        -> DeviceArray
    {
        return CpuDeviceArray(device: self, buffer: buffer)
    }

    //--------------------------------------------------------------------------
	// createQueue
	public func createQueue(name queueName: String,
                             isStatic: Bool) -> DeviceQueue {
        return CpuQueue(logInfo: logInfo.flat(queueName),
                         device: self, name: queueName, isStatic: isStatic)
	}
}


