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

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
import Darwin
#else
import Glibc
#endif

//==============================================================================
/// Executes a closure on the specified queue
/// - Parameter device: the device to set as the single member of the
///  `currentDevices` collection, which is used to execute work
/// - Parameter body: A closure whose operations are to be executed on the
///             specified device
public func using<R>(_ device: ComputeDevice,
                     perform body: () throws -> R) rethrows -> R {
    // sets the default queue and logging info for the current scope
    DeviceContext.local.push(devices: [device])
    defer { DeviceContext.local.popDevices() }
    // execute the body
    return try body()
}

//==============================================================================
/// Executes a closure on the specified queue
/// - Parameter devices: a collection of devices to set as the `currentDevices`
///   collection, which can be used to distribute work
/// - Parameter body: A closure whose operations are to be executed on the
///             specified device
public func using<R>(_ devices: [ComputeDevice],
                     perform body: () throws -> R) rethrows -> R {
    // sets the default queue and logging info for the current scope
    DeviceContext.local.push(devices: devices)
    defer { DeviceContext.local.popDevices() }
    // execute the body
    return try body()
}

//==============================================================================
/// DeviceContext
/// Manages the scope for the current devices, log, and error handlers
@usableFromInline
class DeviceContext {
    /// stack of current device collections used to execute/distribute work
    var devicesStack: [[ComputeDevice]]

    //--------------------------------------------------------------------------
    /// thread data key
    private static let key: pthread_key_t = {
        var key = pthread_key_t()
        pthread_key_create(&key) {
            #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
            let _: AnyObject = Unmanaged.fromOpaque($0).takeRetainedValue()
            #else
            let _: AnyObject = Unmanaged.fromOpaque($0!).takeRetainedValue()
            #endif
        }
        return key
    }()

    //--------------------------------------------------------------------------
    /// returns the thread local instance of the queues stack
    @usableFromInline
    static var local: DeviceContext {
        // try to get an existing state
        if let state = pthread_getspecific(key) {
            return Unmanaged.fromOpaque(state).takeUnretainedValue()
        } else {
            // create and return new state
            let state = DeviceContext()
            pthread_setspecific(key, Unmanaged.passRetained(state).toOpaque())
            return state
        }
    }

    //--------------------------------------------------------------------------
    /// current
    public static var current: [ComputeDevice] {
        return DeviceContext.local.devicesStack.last!
    }

    //--------------------------------------------------------------------------
    /// currentComputeQueue
    // TODO: temporary scheme
    public static var currentComputeQueue: DeviceQueue {
        return DeviceContext.local.devicesStack.last![0].computeQueues[0]
    }
    
    //--------------------------------------------------------------------------
    /// currentTransferQueue
    // TODO: temporary scheme
    public static var currentTransferQueue: DeviceQueue {
        return DeviceContext.local.devicesStack.last![0].transferQueues[0]
    }
    
    //--------------------------------------------------------------------------
    /// hostQueue
    public static var hostQueue: DeviceQueue {
        return DeviceContext.current[0].memory.isUnified ?
            DeviceContext.current[0].transferQueues[0] :
            Platform.local.services[cpuService]!.devices[0].transferQueues[0]
    }

    //--------------------------------------------------------------------------
    /// logInfo
    // `last` is always valid because there will always be
    // the platform default queue and logInfo
    public var logInfo: LogInfo { return devicesStack.last![0].logInfo }

    //--------------------------------------------------------------------------
    // initializers
    private init() {
        devicesStack = [[Platform.local.defaultDevice]]
    }

    //--------------------------------------------------------------------------
    /// push(devices:
    /// pushes the specified device collection onto a queue stack which makes
    /// it the current queue used by operator functions
    @usableFromInline
    func push(devices: [ComputeDevice]) {
        devicesStack.append(devices)
    }

    //--------------------------------------------------------------------------
    /// popDevices
    /// restores the previous current devices collection
    @usableFromInline
    func popDevices() {
        assert(devicesStack.count > 1)
        _ = devicesStack.popLast()
    }
}

////==============================================================================
///// Executes a closure on the specified queue
///// - Parameter queue: the queue to set as the `currentQueue`
///// - Parameter body: A closure whose operations are to be executed on the
/////             specified queue
//public func using<R>(_ queue: DeviceQueue,
//                     perform body: () throws -> R) rethrows -> R {
//    // sets the default queue and logging info for the current scope
//    DeviceContext.local.push(queue: queue)
//    defer { DeviceContext.local.popQueue() }
//    // execute the body
//    return try body()
//}
//
////==============================================================================
///// DeviceContext
///// Manages the scope for the current queue, log, and error handlers
//@usableFromInline
//class DeviceContext {
//    /// stack of default device queues, logging, and exception handler
//    var queueStack: [DeviceQueue]
//
//    //--------------------------------------------------------------------------
//    /// thread data key
//    private static let key: pthread_key_t = {
//        var key = pthread_key_t()
//        pthread_key_create(&key) {
//            #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
//            let _: AnyObject = Unmanaged.fromOpaque($0).takeRetainedValue()
//            #else
//            let _: AnyObject = Unmanaged.fromOpaque($0!).takeRetainedValue()
//            #endif
//        }
//        return key
//    }()
//
//    //--------------------------------------------------------------------------
//    /// returns the thread local instance of the queues stack
//    @usableFromInline
//    static var local: DeviceContext {
//        // try to get an existing state
//        if let state = pthread_getspecific(key) {
//            return Unmanaged.fromOpaque(state).takeUnretainedValue()
//        } else {
//            // create and return new state
//            let state = DeviceContext()
//            pthread_setspecific(key, Unmanaged.passRetained(state).toOpaque())
//            return state
//        }
//    }
//
//    //--------------------------------------------------------------------------
//    /// current
//    public static var current: DeviceQueue {
//        return DeviceContext.local.queueStack.last!
//    }
//
//    //--------------------------------------------------------------------------
//    /// hostQueue
//    public static var hostQueue: DeviceQueue {
//        return DeviceContext.current.device.memory.isUnified ?
//            DeviceContext.current : DeviceContext._umaQueue
//    }
//
//    private static var _umaQueue: DeviceQueue = {
//        // create dedicated queue for app data transfer
//        return try! Platform.local.createQueue(
//            deviceId: 0, serviceName: "cpu", name: "host", isStatic: true)
//    }()
//
//    //--------------------------------------------------------------------------
//    /// auxHostQueue
//    // create dedicated queue for data transfer when accessing
//    // within a queue closure or within HostMultiWrite
//    public static var auxHostQueue: DeviceQueue = {
//        return try! Platform.local.createQueue(
//            deviceId: 0, serviceName: "cpu", name: "dataSync", isStatic: true)
//    }()
//
//    //--------------------------------------------------------------------------
//    /// logInfo
//    // there will always be the platform default queue and logInfo
//    public var logInfo: LogInfo { return queueStack.last!.logInfo }
//
//    //--------------------------------------------------------------------------
//    /// updateDefault
//    public func updateDefault(queue: DeviceQueue) {
//        queueStack[0] = queue
//    }
//
//    //--------------------------------------------------------------------------
//    // initializers
//    private init() {
//        do {
//            // create the default queue based on service and device priority.
//            let queue = try Platform.local.defaultDevice.createQueue(
//                name: "default", isStatic: true)
//            queueStack = [queue]
//        } catch {
//            print("Failed to create default queues")
//            exit(1)
//        }
//    }
//
//    //--------------------------------------------------------------------------
//    /// push(queue:
//    /// pushes the specified queue onto a queue stack which makes
//    /// it the current queue used by operator functions
//    @usableFromInline
//    func push(queue: DeviceQueue) {
//        queueStack.append(queue)
//    }
//
//    //--------------------------------------------------------------------------
//    /// popQueue
//    /// restores the previous current queue
//    @usableFromInline
//    func popQueue() {
//        assert(queueStack.count > 1)
//        _ = queueStack.popLast()
//    }
//}
