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
/// LocalPlatform
/// The default ComputePlatform implementation for a local host
public protocol LocalPlatform : ComputePlatform {
    /// the global services collection
    static var _services: [String: ComputeService]? { get set }
    var _defaultDevice: ComputeDevice? { get set }
    /// a platform wide unique device id obtained during initialization
    static var nextUniqueDeviceId: Int { get }
    /// a platform wide unique stream id obtained during initialization
    static var nextUniqueStreamId: Int { get }
}

public extension LocalPlatform {
    //--------------------------------------------------------------------------
    /// log
    /// the caller can specify a root log which will be inherited by the
    /// device stream hierarchy, but can be overridden at any point down
    /// the tree
    static var log: Log {
        get { return Platform.local.logInfo.log }
        set { Platform.local.logInfo.log = newValue }
    }
    
    //--------------------------------------------------------------------------
    /// handleDevice(error:
    /// The default platform error handler has nowhere else to go, so
    /// print the message, break to the debugger if possible, and exit.
    func handleDevice(error: Error) {
        if (deviceErrorHandler?(error) ?? .propagate) == .propagate {
            print("Unhandled platform error: \(String(describing: error))")
//            raise(SIGINT)
        }
    }

    //--------------------------------------------------------------------------
    // loadServices
    // dynamically loads ComputeService bundles/dylib from the
    // `serviceModuleDirectory` and adds them to the `services` list
    func loadServices() {
        guard Platform._services == nil else { return }
        
        var loadedServices = [String: ComputeService]()
        do {
            // add required cpu service
            let cpuService = try CpuComputeService(platform: Platform.local,
                                                   id: 0, logInfo: logInfo)
            loadedServices[cpuService.name] = cpuService
            
            // add cpu unit test service
            let cpuUnitTestService =
                try CpuUnitTestComputeService(platform: Platform.local,
                                              id: loadedServices.count,
                                              logInfo: logInfo,
                                              name: "cpuUnitTest")
            loadedServices[cpuUnitTestService.name] = cpuUnitTestService

            #if CUDA
            let cudaService = try CudaComputeService(platform: Platform.local,
                                                     id: loadedServices.count,
                                                     logInfo: logInfo)
            loadedServices[cudaService.name] = cudaService
            #endif

            //-------------------------------------
            // dynamically load installed services
            let bundles = getPlugInBundles()
            for bundle in bundles {
                try bundle.loadAndReturnError()
                //            var unloadBundle = false
                
                if let serviceType =
                    bundle.principalClass as? ComputeService.Type {
                    
                    // create the service
                    let service =
                        try serviceType.init(platform: Platform.local,
                                             id: loadedServices.count,
                                             logInfo: logInfo, name: nil)
                    
                    diagnostic(
                        "Loaded compute service '\(service.name)'." +
                        " ComputeDevice count = \(service.devices.count)",
                        categories: .initialize)
                    
                    if service.devices.count > 0 {
                        // add plugin service
                        loadedServices[service.name] = service
                    } else {
                        writeLog("Compute service '\(service.name)' " +
                            "successfully loaded, but reported devices = 0, " +
                            "so service is unavailable", level: .warning)
                        //                    unloadBundle = true
                    }
                }
                // TODO: we should call bundle unload here if there were no devices
                // however simply calling bundle.load() then bundle.unload() making no
                // references to objects inside, later causes an exception in the code.
                // Very strange
                //            if unloadBundle { bundle.unload() }
            }
        } catch {
            writeLog(String(describing: error))
        }
        Platform._services = loadedServices
    }
    
    //--------------------------------------------------------------------------
    /// getPlugInBundles
    /// an array of the dynamically installed bundles
    private func getPlugInBundles() -> [Bundle] {
        if let dir = Bundle.main.builtInPlugInsPath {
            return Bundle.paths(forResourcesOfType: "bundle", inDirectory: dir)
                .map { Bundle(url: URL(fileURLWithPath: $0))! }
        } else {
            return []
        }
    }
    
    //--------------------------------------------------------------------------
    // defaultDevice
    // selects a ComputeDevice based on `servicePriority` and
    // `deviceIdPriority`. It is guaranteed that at least one device like
    // the cpu is available
    var defaultDevice: ComputeDevice {
        guard _defaultDevice == nil else { return _defaultDevice! }
        
        // try to exact match the service request
        let requestedDevice = deviceIdPriority[0]
        for serviceName in servicePriority where _defaultDevice == nil {
            _defaultDevice = requestDevice(serviceName: serviceName,
                                           deviceId: requestedDevice)
        }
        
        // if the search failed, then use the cpu
        _defaultDevice = _defaultDevice ?? requestDevice(serviceName: "cpu")
        // we had to find at least one device like the cpu
        assert(_defaultDevice != nil, "There must be at least one device")

        let device = _defaultDevice!
        writeLog("default device: [\(device.service.name)] \(device.name)",
            level: .status)
        return device
    }
    
    //--------------------------------------------------------------------------
    /// createStream will try to match the requested service name and
    /// device id returning substitutions if needed to fulfill the request
    ///
    /// Parameters
    /// - Parameter deviceId: (0, 1, 2, ...)
    ///   If the specified id is greater than the number of available devices,
    ///   then id % available will be used.
    /// - Parameter serviceName: (cpu, cuda, tpu, ...)
    ///   If no service name is specified, then the default is used.
    /// - Parameter name: a text label assigned to the stream for logging
    func createStream(deviceId id: Int = 0,
                      serviceName: String? = nil,
                      name: String = "stream",
                      isStatic: Bool = false) throws -> DeviceStream {
        
        let serviceName = serviceName ?? defaultDevice.service.name
        if let device = requestDevice(serviceName: serviceName, deviceId: id) {
            return try device.createStream(name: name, isStatic: isStatic)
        } else {
            writeLog("CPU substituted. Service `\(serviceName)` not found.",
                level: .warning)
            let device = requestDevice(serviceName: "cpu")!
            return try device.createStream(name: name, isStatic: isStatic)
        }
    }
    
    //--------------------------------------------------------------------------
    /// requestDevices
    /// - Parameter deviceId: selected device id
    /// - Parameter serviceName: an optional service name to allocate
    ///   the device from.
    /// - Returns: the requested device from the requested service
    ///   substituting if needed based on `servicePriority`
    ///   and `deviceIdPriority`
    func requestDevice(serviceName: String,
                       deviceId: Int = 0) -> ComputeDevice? {
        guard let service = services[serviceName] else { return nil }
        return service.devices[deviceId % service.devices.count]
    }
    
    //--------------------------------------------------------------------------
    /// open
    /// this is a placeholder. Additional parameters will be needed for
    /// credentials, timeouts, etc...
    ///
    /// - Parameter url: the location of the remote platform
    /// - Returns: a reference to the remote platform, which can be used
    ///   to query resources and create remote streams.
    static func open(platform url: URL) throws -> ComputePlatform {
        fatalError("not implemented yet")
    }
}

//==============================================================================
// Platform
/// The root object to select compute services and devices
final public class Platform: LocalPlatform {
    // properties
    public var _defaultDevice: ComputeDevice?
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _errorMutex: Mutex = Mutex()
    public var _lastError: Error? = nil
    public var deviceIdPriority: [Int] = [0]
    public var id: Int = 0
    public static let local = Platform()
    public var serviceModuleDirectory = URL(fileURLWithPath: "TODO")
    public var servicePriority = ["cuda", "cpu"]
    public lazy var services: [String : ComputeService] = {
        loadServices()
        return Platform._services!
    }()
    public static var _services: [String: ComputeService]?
    public private(set) var trackingId = 0
    public var logInfo: LogInfo

    /// a platform wide unique stream id obtained during initialization
    private static var deviceIdCounter = AtomicCounter(value: -1)
    public static var nextUniqueDeviceId: Int {
        return Platform.deviceIdCounter.increment()
    }

    /// a platform wide unique stream id obtained during initialization
    private static var streamIdCounter = AtomicCounter(value: -1)
    public static var nextUniqueStreamId: Int {
        return Platform.streamIdCounter.increment()
    }

    //--------------------------------------------------------------------------
    // initializers
    /// `init` is private because this is a singleton. Use the `local` static
    /// member to access the shared instance.
    private init() {
        // create the log
        logInfo = LogInfo(log: Log(isStatic: true), logLevel: .error,
                          namePath: String(describing: Platform.self),
                          nestingLevel: 0)
    }
}
