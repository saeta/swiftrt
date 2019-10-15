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
// Platform Abstraction Protocols
//
//  ComputePlatform
//      services[]
//        ComputeService (cpu, vulkan, cuda, ...)
//          devices[]
//            ComputeDevice (dev:0, dev:1, ...)
//              DeviceArray
//              DeviceQueue
//                CommandBuffer
//                QueueEvent
//
import Foundation

//==============================================================================
/// ComputePlatform
/// The compute platform is the root object for managing all services, devices,
/// and queues. There is one local instance per process, and possibly
/// many remote instances.
public protocol ComputePlatform: DeviceErrorHandling, ObjectTracking, Logger {
    /// global shared instance
    static var local: Platform { get }
    /// the root log
    static var log: Log { get set }

    // instance members
    /// a device automatically selected based on service priority
    var defaultDevice: ComputeDevice { get }
    /// ordered list of device ids specifying the order for auto selection
    var deviceIdPriority: [Int] { get set }
    /// the platform id. Usually zero, but can be assigned in case a higher
    /// level object (e.g. cluster) will maintain a platform collection
    var id: Int { get set }
    /// location of dynamically loaded service modules
    var serviceModuleDirectory: URL { get set }
    /// ordered list of service names specifying the order for auto selection
    var servicePriority: [String] { get set }
    /// dictionary of service configuration property sets
    /// Key is service name, value is serive property set
    /// Property set [property name : any type specific value]
    static var serviceProperties: [String : [String : Any]] { get set }
    /// a dynamically loaded collection of available compute services.
    /// The "cpu" service will always be available
    var services: [String : ComputeService] { get }
    
    //--------------------------------------------------------------------------
    /// createQueue will try to match the requested service name and
    /// device id returning substitutions if needed to fulfill the request
    ///
    /// Parameters
    /// - Parameter deviceId: (0, 1, 2, ...)
    ///   If the specified id is greater than the number of available devices,
    ///   then id % available will be used.
    /// - Parameter serviceName: (cpu, cuda, tpu, ...)
    ///   If no service name is specified, then the default is used.
    /// - Parameter name: a text label assigned to the queue for logging
    /// - Parameter isStatic: if `true` the object will not be reported
    ///   as a memory leak
    func createQueue(deviceId: Int,
                     serviceName: String?,
                     name: String,
                     isStatic: Bool) throws -> DeviceQueue
    
    //--------------------------------------------------------------------------
    /// requestDevices
    /// - Parameter serviceName: the service to allocate the device from.
    /// - Parameter deviceId: selected device id
    /// - Returns: the requested device from the requested service
    ///   substituting if needed based on `servicePriority`
    ///   and `deviceIdPriority`
    func requestDevice(serviceName: String, deviceId: Int) -> ComputeDevice?
}

//==============================================================================
/// ComputeService
/// a compute service represents category of installed devices on the platform,
/// such as (cpu, cuda, tpu, ...)
public protocol ComputeService: ObjectTracking, Logger, DeviceErrorHandling {
    /// a collection of available devices
    var devices: [ComputeDevice] { get }
    /// the service id
    var id: Int { get }
    /// the service name used for `servicePriority` and logging
    var name: String { get }
    /// the platform this service belongs to
    var platform: ComputePlatform! { get }
    /// The default maximum amount of time allowed for an operation to complete.
    /// `timeout` is inherited by devices and queues when they are created.
    var timeout: TimeInterval? { get set }

    /// required initializer to support dynamically loaded services
    /// - Parameter platform: the parent platform object
    /// - Parameter id: the service id
    /// - Parameter logInfo: the log information to use
    /// - Parameter name: an optional service name
    init(platform: ComputePlatform,
         id: Int,
         logInfo: LogInfo,
         name: String?) throws
}

//==============================================================================
/// ServiceError
/// errors thrown from a ComputeService
public enum ServiceError : Error {
    case serviceIsUnavailable
    case functionFailure(location: String, message: String)
    case rangeError(String)
}

//==============================================================================
/// LocalComputeService
public protocol LocalComputeService: ComputeService { }

public extension LocalComputeService {
    //--------------------------------------------------------------------------
    /// handleDevice(error:
    func handleDevice(error: Error) {
        if (deviceErrorHandler?(error) ?? .propagate) == .propagate {
            platform.handleDevice(error: error)
        }
    }
}

//==============================================================================
/// ComputeDevice
/// a compute device represents a physical service device installed
/// on the platform
public protocol ComputeDevice: ObjectTracking, Logger, DeviceErrorHandling {
    //-------------------------------------
    // properties
    /// a key to lookup device array replicas
    var deviceArrayReplicaKey: Int { get }
    /// parameters defining maximum device capabilties
    var limits: DeviceLimits { get }
    /// the id of the device for example dev:0
    var id: Int { get }
    /// the name of the device
    var name: String { get }
    /// the service this device belongs to
    var service: ComputeService! { get }
    /// the maximum amount of time allowed for an operation to complete
    var timeout: TimeInterval? { get set }
    /// the type of memory addressing this device uses
    var memoryAddressing: MemoryAddressing { get }

    //-------------------------------------
    // device resource functions
    /// creates an array on this device
    func createArray(count: Int) throws -> DeviceArray
    /// creates a device array from a uma buffer.
    func createReferenceArray(buffer: UnsafeRawBufferPointer) -> DeviceArray
    /// creates a device array from a uma buffer.
    func createMutableReferenceArray(buffer: UnsafeMutableRawBufferPointer)
        -> DeviceArray
    /// creates a named command queue for this device
    /// - Parameter isStatic: if `true` the object will not be tracked
    func createQueue(name: String, isStatic: Bool) throws -> DeviceQueue
}

public enum MemoryAddressing { case unified, discreet }

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
/// LocalComputeDevice
public protocol LocalComputeDevice: ComputeDevice { }

public extension LocalComputeDevice {
    //--------------------------------------------------------------------------
    /// handleDevice(error:
    func handleDevice(error: Error) {
        if (deviceErrorHandler?(error) ?? .propagate) == .propagate {
            service.handleDevice(error: error)
        }
    }
}

//==============================================================================
// DeviceArray
//    This represents a device data array
public protocol DeviceArray: ObjectTracking {
    /// a pointer to the memory on the device
    var buffer: UnsafeMutableRawBufferPointer { get }
    /// the device that created this array
    var device: ComputeDevice { get }
    /// `true` if the array is read only
    var isReadOnly: Bool { get }
    /// the array edit version number used for replication and synchronization
    var version: Int { get set }
}

//==============================================================================
/// QueueEvent
/// A queue event is a barrier synchronization object that is
/// - created by a `ComputeDevice`
/// - recorded on a queue to create a barrier
/// - waited on by one or more threads for group synchronization
public protocol QueueEvent: ObjectTracking {
    /// is `true` if the even has occurred, used for polling
    var occurred: Bool { get }
    /// the last time the event was recorded
    var recordedTime: Date? { get set }
    /// measure elapsed time since another event
    func elapsedTime(since other: QueueEvent) -> TimeInterval?
    /// will block the caller until the timeout has elapsed if one
    /// was specified during init, otherwise it will block forever
    func wait() throws
}

public extension QueueEvent {
    //--------------------------------------------------------------------------
    /// elapsedTime
    /// computes the timeinterval between two queue event recorded times
    /// - Parameter other: the other event used to compute the interval
    /// - Returns: the elapsed interval. Will return `nil` if this event or
    ///   the other have not been recorded.
    func elapsedTime(since other: QueueEvent) -> TimeInterval? {
        guard let time = recordedTime,
              let other = other.recordedTime else { return nil }
        return time.timeIntervalSince(other)
    }
}

public struct QueueEventOptions: OptionSet {
    public init() { self.rawValue = 0 }
    public init(rawValue: Int) { self.rawValue = rawValue }
    public let rawValue: Int
    public static let timing       = QueueEventOptions(rawValue: 1 << 0)
    public static let interprocess = QueueEventOptions(rawValue: 1 << 1)
}

public enum QueueEventError: Error {
    case timedOut
}
