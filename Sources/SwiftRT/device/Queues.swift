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
/// - Parameter queue: the queue to set as the `currentQueue`
/// - Parameter body: A closure whose operations are to be executed on the
///             specified queue
public func using<R>(_ queue: DeviceQueue,
                     perform body: () throws -> R) rethrows -> R {
    // sets the default queue and logging info for the current scope
    _Queues.local.push(queue: queue)
    defer { _Queues.local.popQueue() }
    // execute the body
    return try body()
}

//==============================================================================
/// _Queues
/// Manages the scope for the current queue, log, and error handlers
@usableFromInline
class _Queues {
    /// stack of default device queues, logging, and exception handler
    var queueStack: [DeviceQueue]

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
    static var local: _Queues {
        // try to get an existing state
        if let state = pthread_getspecific(key) {
            return Unmanaged.fromOpaque(state).takeUnretainedValue()
        } else {
            // create and return new state
            let state = _Queues()
            pthread_setspecific(key, Unmanaged.passRetained(state).toOpaque())
            return state
        }
    }

    //--------------------------------------------------------------------------
    /// current
    public static var current: DeviceQueue {
        return _Queues.local.queueStack.last!
    }
    
    //--------------------------------------------------------------------------
    /// hostQueue
    public static var hostQueue: DeviceQueue {
        return _Queues.current.device.memoryAddressing == .unified ?
            _Queues.current : _Queues._umaQueue
    }
    
    private static var _umaQueue: DeviceQueue = {
        // create dedicated queue for app data transfer
        return try! Platform.local.createQueue(
            deviceId: 0, serviceName: "cpu", name: "host", isStatic: true)
    }()
    
    //--------------------------------------------------------------------------
    /// auxHostQueue
    // create dedicated queue for data transfer when accessing
    // within a queue closure or within HostMultiWrite
    public static var auxHostQueue: DeviceQueue = {
        return try! Platform.local.createQueue(
            deviceId: 0, serviceName: "cpu", name: "dataSync", isStatic: true)
    }()
    
    //--------------------------------------------------------------------------
    /// logInfo
    // there will always be the platform default queue and logInfo
    public var logInfo: LogInfo { return queueStack.last!.logInfo }

    //--------------------------------------------------------------------------
    /// updateDefault
    public func updateDefault(queue: DeviceQueue) {
        queueStack[0] = queue
    }
    
    //--------------------------------------------------------------------------
    // initializers
    private init() {
        do {
            // create the default queue based on service and device priority.
            let queue = try Platform.local.defaultDevice.createQueue(
                    name: "default", isStatic: true)
            queueStack = [queue]
        } catch {
            print("Failed to create default queues")
            exit(1)
        }
    }
    
    //--------------------------------------------------------------------------
    /// push(queue:
    /// pushes the specified queue onto a queue stack which makes
    /// it the current queue used by operator functions
    @usableFromInline
    func push(queue: DeviceQueue) {
        queueStack.append(queue)
    }
    
    //--------------------------------------------------------------------------
    /// popQueue
    /// restores the previous current queue
    @usableFromInline
    func popQueue() {
        assert(queueStack.count > 1)
        _ = queueStack.popLast()
    }
}
