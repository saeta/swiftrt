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
    public private(set) var computeQueues = [DeviceQueue]()
    public private(set) var transferQueues = [DeviceQueue]()
    public let deviceArrayReplicaKey = Platform.nextUniqueDeviceId
    public let id: Int
    public var logInfo: LogInfo
    public let name: String
    public weak var service: ComputeService!
    public var timeout: TimeInterval?
    public var deviceErrorHandler: DeviceErrorHandler?
    public var limits: DeviceLimits
    public var memory: MemoryProperties
    public var _lastError: Error? = nil
    public var _errorMutex: Mutex = Mutex()
    
    // configuration and defaults
    public var configuration: [CudaPropertyKey: Any] = [
        .queuesPerDevice: 2
    ]
    
    //--------------------------------------------------------------------------
	// initializers
	public init(service: CpuService,
                deviceId: Int,
                logInfo: LogInfo,
                isUnified: Bool,
                timeout: TimeInterval?) {
        self.name = "cpu:\(deviceId)"
		self.logInfo = logInfo.flat("cpu:\(deviceId)")
		self.id = deviceId
		self.service = service
        self.timeout = timeout
        
        // TODO: determine meaningful values, not currently used
        self.limits = DeviceLimits(
            maxComputeSharedMemorySize: 1,
            maxComputeWorkGroupCount: (1, 1, 1),
            maxComputeWorkGroupInvocations: 1,
            maxComputeWorkGroupSize: (1, 1, 1),
            maxMemoryAllocationCount: 1
        )
        
        // TODO:
        self.memory = MemoryProperties(isUnified: isUnified,
                                       heaps: [MemoryHeap]())

        //---------------------------------
        // create device queues
        assert(service.configuration[.queuesPerDevice] is Int)
        let queueCount = service.configuration[.queuesPerDevice] as! Int
        var queues = [DeviceQueue]()
        for id in 0..<queueCount {
            let queueName = "queue:\(id)"
            queues.append(CpuQueue(logInfo: logInfo.flat(queueName),
                                   device: self, name: queueName,
                                   id: Platform.nextUniqueQueueId,
                                   isStatic: true))
        }
        computeQueues = queues
        transferQueues = computeQueues

		// devices are statically held by the Platform.service
        trackingId = ObjectTracker.global
            .register(self, namePath: logNamePath, isStatic: true)
	}
	deinit { ObjectTracker.global.remove(trackingId: trackingId) }

    //--------------------------------------------------------------------------
	// createArray
	//	This creates memory on the device
    public func createArray(count: Int, heapIndex: Int = 0) throws
        -> DeviceArray
    {
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
}
