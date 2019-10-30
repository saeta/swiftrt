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
public struct DeviceContextState {
    /// current collection of devices used to execute/distribute work
    var devices: [ComputeDevice]
    /// specifies whether operators in the current scope are
    /// evaluated for training or inferring
    var evaluationMode: EvaluationMode = .inferring
    /// helpers
    var isInferring: Bool { return evaluationMode == .inferring }
    var isTraining: Bool { return evaluationMode == .training }
}

//==============================================================================
/// DeviceContext
/// Manages the scope for the current devices, log, and error handlers
@usableFromInline
class DeviceContext {
    /// stack of current device collections used to execute/distribute work
    var stateStack: [DeviceContextState]
    
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
    public static var current: DeviceContextState {
        return DeviceContext.local.stateStack.last!
    }

    //--------------------------------------------------------------------------
    /// currentQueue
    // TODO: temporary scheme
    public static var currentQueue: DeviceQueue {
        return DeviceContext.local.stateStack.last!.devices[0].queues[0]
    }
    
    //--------------------------------------------------------------------------
    /// hostQueue
    public static var hostQueue: DeviceQueue {
        let currentDevice = DeviceContext.current.devices[0]
        return currentDevice.memory.addressing == .unified ?
            currentDevice.queues[0] : Platform.cpu.queues[0]
    }

    //--------------------------------------------------------------------------
    /// logInfo
    // `last` is always valid because there will always be
    // the platform default queue and logInfo
    public var logInfo: LogInfo { return stateStack.last!.devices[0].logInfo }

    //--------------------------------------------------------------------------
    // initializers
    private init() {
        stateStack = [
            DeviceContextState(devices: [Platform.local.defaultDevice],
                               evaluationMode: .inferring)
        ]
    }

    //--------------------------------------------------------------------------
    /// push(devices:
    /// pushes the specified device collection onto a queue stack which makes
    /// it the current queue used by operator functions
    @usableFromInline
    func push(devices: [ComputeDevice]) {
        let evaluationMode = DeviceContext.local.stateStack.last!.evaluationMode
        let state = DeviceContextState(devices: devices,
                                       evaluationMode: evaluationMode)
        stateStack.append(state)
    }

    //--------------------------------------------------------------------------
    /// popDevices
    /// restores the previous current devices collection
    @usableFromInline
    func popDevices() {
        assert(stateStack.count > 1)
        _ = stateStack.popLast()
    }
}
