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

//==============================================================================
// VulkanComputeService
public final class VulkanComputeService: LocalComputeService {
    // service protocol properties
    public private(set) weak var platform: ComputePlatform!
    public private(set) var trackingId = 0
    public private(set) var devices = [ComputeDevice]()
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error?
    public var _errorMutex: Mutex = Mutex()
    public let id: Int
    public var logInfo: LogInfo
    public let name: String

    // vulkan specific properties
    private var instance: VkInstance!
    
    //--------------------------------------------------------------------------
    // timeout
    public var timeout: TimeInterval? {
        didSet {
            devices.forEach {
                $0.timeout = timeout
            }
        }
    }

    //--------------------------------------------------------------------------
    // initializers
    public required init(platform: ComputePlatform,
                         id: Int,
                         logInfo: LogInfo,
                         name: String? = nil) throws {
        self.platform = platform
        self.id = id
        self.name = name ?? "vulkan"
        self.logInfo = logInfo
        
        // create the vulkan instance
        instance = try createVkInstance()
        
        // this is held statically by the Platform
        trackingId = ObjectTracker.global.register(self, isStatic: true)
    }

    deinit {
        ObjectTracker.global.remove(trackingId: trackingId)
    }

    //--------------------------------------------------------------------------
    // createVkInstance
    // Creates the VkInstance object, which is the root of the environment
    private func createVkInstance() throws -> VkInstance {
        // list of enabled layers for validation and extensions
        var enabledLayers = [CStringPointer?]()
        var enabledExtensions = [CStringPointer?]()
        
        //----------------------------------------------------------------------
        // Enable Vulkan validation layers in DEBUG mode
        #if DEBUG
        // query supported validation layers to find "standard_validation"
        var layerCount: UInt32 = 0
        try vkCheck(vkEnumerateInstanceLayerProperties(&layerCount, nil))

        var layerProps = [VkLayerProperties](repeating: VkLayerProperties(),
                                             count: Int(layerCount))
        try vkCheck(vkEnumerateInstanceLayerProperties(&layerCount,
                                                       &layerProps))
        
        let stdName = "VK_LAYER_LUNARG_standard_validation"
        for i in 0..<layerProps.count {
            // point to retained string tuple
            let layerNamePointer = withUnsafeBytes(of: &layerProps[i].layerName) {
                return $0.baseAddress!.assumingMemoryBound(to: CChar.self)
            }
            if String(cString: layerNamePointer) == stdName {
                print(String(cString: layerNamePointer))
                enabledLayers.append(layerNamePointer)
                break
            }
        }
        if enabledLayers.isEmpty {
            writeLog("Layer \(stdName) not supported", level: .warning)
        }
        
        // Enable VK_EXT_DEBUG_REPORT_EXTENSION_NAME extension so that
        // validation layers will emit warnings
        var extensionCount: UInt32 = 0
        try vkCheck(vkEnumerateInstanceExtensionProperties(nil,
                                                           &extensionCount,
                                                           nil))
        var extensionProps =
            [VkExtensionProperties](repeating: VkExtensionProperties(),
                                    count: Int(extensionCount))
        try vkCheck(vkEnumerateInstanceExtensionProperties(nil,
                                                           &extensionCount,
                                                           &extensionProps))
        let extName = VK_EXT_DEBUG_REPORT_EXTENSION_NAME
        for i in 0..<extensionProps.count {
            // point to retained string tuple
            let extensionNamePointer =
                withUnsafeBytes(of: &extensionProps[i].extensionName) {
                return $0.baseAddress!.assumingMemoryBound(to: CChar.self)
            }
            
            if String(cString: extensionNamePointer) == extName {
                print(String(cString: extensionNamePointer))
                enabledExtensions.append(extensionNamePointer)
                break
            }
        }
        if enabledExtensions.isEmpty {
            writeLog("Extension \(extName) not supported", level: .warning)
        }
        #endif

        //----------------------------------------------------------------------
        // initialize the VkApplicationInfo.
        // TODO: revisit to determine what the meaningful values should be
        var applicationInfo = VkApplicationInfo(
            sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext: nil,
            pApplicationName: nil,
            applicationVersion: 0,
            pEngineName: nil,
            engineVersion: 0,
            apiVersion: UInt32(VK_VERSION_1_1))
        
        var createInfo = VkInstanceCreateInfo(
            sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pNext: nil,
            flags: 0,
            pApplicationInfo: &applicationInfo,
            enabledLayerCount: UInt32(enabledLayers.count),
            ppEnabledLayerNames: &enabledLayers,
            enabledExtensionCount: UInt32(enabledExtensions.count),
            ppEnabledExtensionNames: &enabledExtensions)
        
        // create the instance. It will throw if it failed, so it can't be nil.
        var instance: VkInstance?
        try vkCheck(vkCreateInstance(&createInfo, nil, &instance))
        return instance!
    }
}
